#Ground Truth

class DPCKNNExpand(Operation):
    def __init__(
            self,
            num_clusters: int = 4,
            k: int = 5,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            use_density_for_score: bool = True,
            use_density_minimum_as_border: bool = False,
            reset_seed: bool = False
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.k = k
        self.clip_model_name = clip_model_name
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        # treat the expansion as an api function regarding seed reset since it is sensitive to non-deterministic behavior
        # this might cause conflicts with summaries that have been extracted before this seed reset was added,
        # that's why it can be controlled by the hyperparameter reset_seed (defaults to False since that was the
        # behavior before the seed reset was added)
        if self.reset_seed:
            api.reset_seed()

        # get the state of the source node
        source_node_state = target.state

        # get the video clip of the state of the source node
        source_video_clip = source_node_state.video_clip

        video = source_video_clip.data

        # the following cluster center selection using DPC KNN is inspired by:
        # https://github.com/PKU-YuanGroup/Chat-UniVi
        # 1. get image features
        encoder = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = encoder(images=video, return_tensors="pt")
        video_features = inputs["pixel_values"]
        logger.debug(f"video_features.shape: {video_features.shape}")
        # num_frames, num_channels, 224, 224

        # 2. transform image features to token features, i.e. flatten to patches of size 1024
        features = video_features.view(video.size(0), -1, 1024)
        logger.debug(f"features.shape: {features.shape}")
        # num_frames, (num_channels * 224 * 224) / 1024, 1024

        # 3. get mean along the second dimension, i.e. mean pooling
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
        token_dict = {'x': cls_features,
                      'token_num': cls_features.size(1),
                      'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),
                      'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),
                      'mask': None}
        logger.debug(f"cls_features.shape: {cls_features.shape}")
        logger.debug(f"token_dict['idx_token'].shape: {token_dict['idx_token'].shape}")
        logger.debug(f"token_dict['agg_weight'].shape: {token_dict['agg_weight'].shape}")

        # 4. cluster tokens with DPC-KNN algorithm
        k = len(source_video_clip) if len(source_video_clip) < self.k else self.k
        num_clusters = len(source_video_clip) if len(source_video_clip) < self.num_clusters else self.num_clusters
        idx_cluster, cluster_num, centers, cluster_borders = DPCKNNExpand.cluster_dpc_knn(token_dict,
                                                                                          num_clusters,
                                                                                          k,
                                                                                          token_mask=token_dict["mask"],
                                                                                          use_density_for_score=self.use_density_for_score)
        logger.debug(f"idx_cluster: {idx_cluster}")
        logger.debug(f"cluster_num: {cluster_num}")
        logger.debug(f"cluster_borders: {cluster_borders}")

        # get the cluster centers and sort them in ascending order
        centers = centers[0].tolist()
        centers.sort()
        logger.debug(f"centers: {centers}")

        # remove the first frame from the centers since they will be processed anyway in the next steps
        centers.remove(0) if 0 in centers else None
        logger.debug(f"centers after removing first frame: {centers}")

        # get the borders between the clusters
        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video.size(0))
        else:
            # get the borders of the clusters (middle between two centers)
            for i in range(len(centers)):
                if i == len(centers) - 1:
                    borders.append(video.size(0))
                else:
                    # floor the middle between two centers, such that the last frame is still a valid index
                    border = int(math.floor((centers[i + 1] - centers[i]) / 2 + centers[i]))
                    borders.append(border)
        logger.debug(f"boundaries (borders): {borders}")

        target_nodes = []
        for i in range(1, len(borders)):
            # get the list indices for the new video clip
            start_list_index = borders[i - 1]
            end_list_index = borders[i]
            logger.debug(f"start_index (relative) of new video clip: {start_list_index}")
            logger.debug(f"end_index (relative) of new video clip: {end_list_index}")

            # trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]

            # trim the sampled indices to the split
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

            # create a new video clip from the split
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            # create a new node with new state for the new node
            new_node = Node(
                state=NodeState(
                    video_clip=split_video_clip,
                    task=source_node_state.task,
                    lexical_representation=source_node_state.lexical_representation,
                    spatial_node_state=SpatialNodeState(
                        video_clip=split_video_clip,
                        task=source_node_state.spatial_node_state.task,
                        use_action_captions=source_node_state.spatial_node_state.use_action_captions,
                        use_object_detections=source_node_state.spatial_node_state.use_object_detections,
                        use_action_captions_summary=source_node_state.spatial_node_state.use_action_captions_summary,
                        use_object_detections_summary=source_node_state.spatial_node_state.use_object_detections_summary,
                        lexical_representation=source_node_state.spatial_node_state.lexical_representation
                    ),
                    temporal_node_state=TemporalNodeState(
                        video_clip=split_video_clip,
                        task=source_node_state.temporal_node_state.task,
                        use_temporal_grounding_summary=source_node_state.temporal_node_state.use_temporal_grounding_summary,
                        lexical_representation=source_node_state.temporal_node_state.lexical_representation,
                        use_relevance=source_node_state.temporal_node_state.use_relevance,
                        use_salience=source_node_state.temporal_node_state.use_salience,
                        use_foreground=source_node_state.temporal_node_state.use_foreground
                    )
                )
            )

            target_nodes.append(new_node)

        # apply the expansion to the graph
        edges = [Edge(source=source_node, target_node=target_node)
                 for target_node in target_nodes
                 for source_node in [target]]

        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

        logger.info(f"Executed structure expansion operation: DPCKNNExpand")

    @staticmethod
    def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None, use_density_for_score=True):
        """Cluster tokens with DPC-KNN algorithm.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
            use_density_for_score (bool): whether to use density to compute the score or not.
        """
        with torch.no_grad():
            x = token_dict["x"]
            B, N, C = x.shape
            logger.debug(f"x.shape: {x.shape}")

            # compute pairwise distance matrix
            dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            logger.debug(f"dist_nearest: {dist_nearest.shape}")
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            logger.debug(f"density: {density.shape}")

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            # (case distinction is added by maximotus, not part of the original implementation)
            score = dist * density if use_density_for_score else dist
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            logger.debug(f"score: {score}")
            logger.debug(f"index_down: {index_down}")

            # assign tokens to the nearest center
            dist_matrix = DPCKNNExpand.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

            # make list from index_down tensor
            index_down_list = index_down.tolist()[0]
            index_down_list.sort()
            logger.debug(f"index_down_list: {index_down_list}")

            # get the frame indices of the density minimums between cluster centers as borders
            # (this is added by maximotus, not part of the original implementation)
            borders = []
            for j in range(cluster_num - 1):
                # get the current and next cluster center indices
                current_cluster_center_index = index_down_list[j]
                next_cluster_center_index = index_down_list[j + 1]

                # slice the density tensor to get the density values
                # between the current cluster center and the next cluster center (excluding both)
                density_slice = density[:, current_cluster_center_index + 1:next_cluster_center_index - 1]
                logger.debug(f"density_slice: {density_slice.shape}")

                # get the frame index of the minimum density value
                if density_slice.size(1) == 0:
                    min_density_idx = current_cluster_center_index + 1
                else:
                    min_density_idx = density_slice.argmin(dim=1).item() + current_cluster_center_index + 1
                logger.debug(f"min_density_idx: {min_density_idx}")

                # add the frame index of the minimum density value to the borders list
                borders.append(min_density_idx)

        return idx_cluster, cluster_num, index_down, borders

    @staticmethod
    def index_points(points, idx):
        """Sample features following the index.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def __str__(self):
        return f"DPCKNNExpand(num_clusters={self.num_clusters}, k={self.k}, clip_model_name={self.clip_model_name})"
############################################################################################################## 
# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# Please help me to write class DPCKNNExpand , which inherits Operation baseclass. it takes  num_clusters: int = 4,k: int = 5,clip_model_name: str = "openai/clip-vit-large-patch14",use_density_for_score: bool = True,use_density_minimum_as_border: bool = False,reset_seed: bool = False as inputs
# then initialize member variables for self.num_clusters,self.clip_model_name,self.k,self.use_density_for_score,self.use_density_minimum_as_border,self.reset_seed
# the first function of class is _execute it takes graph: Optional[Graph], api: Optional[API], target: Optional[Node] as inputs and returns None
# if self.reset_seed is true then call api.reset_seed(), then get source_node_state from target.state . and similarly get the video clip of the state of the source node
# then firstly using CLIPImageProcessor.from_pretrained(self.clip_model_name) to get image encoder and get inputs variable from encoder(images=video, return_tensors="pt")
# extract pixel_values from inputs variables to video_features
# then transform image feature to token feature and flatten to patches of size 1024 like  video_features.view(video.size(0), -1, 1024)
# logger.debug the shape of processed results get mean along the second dimension, i.e. mean pooling for feratures and assign it to cls_features here using torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
# generate the dictionary token_dict from cls_features : 'x': cls_features,'token_num': cls_features.size(1),'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),'mask': None
# and then cluster tokens with DPC-KNN algorith
# if len(source_video_clip) < self.k then set k = len(source_video_clip) ohterwise set k to self.k
# if len(source_video_clip) < self.num_clusters  then num_clusters = len(source_video_clip) ohterwise set it as self.num_clusters
# here then call DPCKNNExpand.cluster_dpc_knn function which takes token_dict,num_clusters,k, token_mask and use_density_for_score  as inputs
# the cluster_dpc_knn will returns and assign some variables: idx_cluster, cluster_num, centers, cluster_borders
# get the cluster centers and sort them in ascending order and then emove the first frame from the centers 
# then initialize borders as [0] list , and if self.use_density_minimum_as_border is true then run borders.extend(cluster_borders) and borders.append(video.size(0))
# if is false then iterate in len(centers) if i is i == len(centers) - 1 then borders.append(video.size(0)), else loor the middle between two centers and set it to border. then append border to borders
# initialize the tartget_nodes as []. next get the list indices for the new video clip and  trim the video data to the split and  trim the sampled indices to the split  and then create a new video clip from the split. Here respectively generating two variables:split_data and split_sampled_indices
# then depending on the generated inputs to initialize  split_video_clip a instance of VideoClip and then based on the split_video_clip to create a new node with new state for the new node calling Node class needing Nodestate and SpatialNodeState and TemporalNodeState
# the new instance for Node will be appended into target_node list. 
# next apply the expansion to the Edges and graph these two variables, The expansion way of Edges is [Edge(source=source_node, target_node=target_node) for target_node in target_nodes for source_node in [target]]
# while graph can directly call its function add_nodes and add_edges
# the second function is named cluster_dpc_knn. It's also a staticmethod. It will take token_dict, cluster_num k=5 , token_mask=None and use_density_for_score=True as its inputs
# and return idx_cluster,index_down, borders 
# the next function is a staticmethod named index_points which takes points and idx as inputs and return new_points , it can sample features following the index
# the last function of the class is __str__ , it will return elf.num_clusters,self.k,self.clip_model_name
class DPCKNNExpand(Operation):
    def __init__(
        self,
        num_clusters: int = 4,
        k: int = 5,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        use_density_for_score: bool = True,
        use_density_minimum_as_border: bool = False,
        reset_seed: bool = False
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.k = k
        self.clip_model_name = clip_model_name
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node]
    ) -> None:
        if target is None or api is None or graph is None:
            logger.error("Graph, API, and target Node must be provided.")
            return

        if self.reset_seed:
            logger.debug("Resetting seed.")
            api.reset_seed()

        source_node_state: NodeState = target.state
        source_video_clip: VideoClip = source_node_state.video_clip
        video: torch.Tensor = source_video_clip.data  # Assuming video is a tensor

        logger.debug(f"Original video shape: {video.shape}")

        # Initialize CLIP Image Processor and Encoder
        image_processor = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = image_processor(images=video, return_tensors="pt")
        video_features = inputs['pixel_values']

        logger.debug(f"Video features shape: {video_features.shape}")

        # Transform image features to token features
        features = video_features.view(video.size(0), -1, 1024)
        logger.debug(f"Transformed features shape: {features.shape}")

        # Mean pooling
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
        logger.debug(f"CLS features shape after mean pooling: {cls_features.shape}")

        # Generate token dictionary
        token_dict = {
            'x': cls_features,
            'token_num': cls_features.size(1),
            'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),
            'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),
            'mask': None
        }

        # Adjust k and num_clusters based on video length
        video_length = source_video_clip.data.size(0)
        k = min(self.k, video_length)
        num_clusters = min(self.num_clusters, video_length)

        logger.debug(f"Using k={k} and num_clusters={num_clusters} based on video length={video_length}")

        # Cluster tokens using DPC-KNN
        idx_cluster, cluster_num, centers, cluster_borders = self.cluster_dpc_knn(
            token_dict=token_dict,
            cluster_num=num_clusters,
            k=k,
            token_mask=None,
            use_density_for_score=self.use_density_for_score
        )

        # Sort centers and remove the first frame
        centers_sorted, _ = torch.sort(centers)
        centers_sorted = centers_sorted[1:]
        logger.debug(f"Sorted centers (excluding first frame): {centers_sorted}")

        # Initialize borders
        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video.size(0))
        else:
            for i in range(len(centers_sorted)):
                if i == len(centers_sorted) - 1:
                    border = video.size(0)
                else:
                    border = (centers_sorted[i] + centers_sorted[i + 1]) // 2
                borders.append(border)
        logger.debug(f"Final borders: {borders}")

        target_nodes = []
        for i in range(len(borders) - 1):
            split_data = video[borders[i]:borders[i + 1]]
            split_sampled_indices = torch.arange(borders[i], borders[i + 1])

            split_video_clip = VideoClip(data=split_data, sampled_indices=split_sampled_indices)

            # Create new NodeState (assuming default spatial and temporal states)
            new_state = NodeState(
                video_clip=split_video_clip,
                spatial_state=SpatialNodeState(),
                temporal_state=TemporalNodeState()
            )

            new_node = Node(state=new_state)
            target_nodes.append(new_node)
            logger.debug(f"Created new node with video clip from {borders[i]} to {borders[i + 1]}")

        # Add new nodes to the graph
        graph.add_nodes(target_nodes)
        logger.debug(f"Added {len(target_nodes)} new nodes to the graph.")

        # Create edges between source node and target nodes
        edges = [Edge(source=target, target_node) for target_node in target_nodes]
        graph.add_edges(edges)
        logger.debug(f"Added {len(edges)} new edges to the graph.")

    @staticmethod
    def cluster_dpc_knn(
        token_dict: Dict[str, Any],
        cluster_num: int,
        k: int = 5,
        token_mask: Optional[torch.Tensor] = None,
        use_density_for_score: bool = True
    ) -> Any:
        """
        Placeholder for the DPC-KNN clustering algorithm.
        Replace this with the actual implementation.
        """
        # Example dummy implementation
        idx_cluster = torch.randint(0, cluster_num, (token_dict['x'].size(1),))
        cluster_num_returned = cluster_num
        centers = torch.randint(0, token_dict['x'].size(1), (cluster_num,))
        cluster_borders = torch.randint(0, token_dict['x'].size(1), (cluster_num - 1,)).tolist()
        logger.debug("Performed DPC-KNN clustering.")
        return idx_cluster, cluster_num_returned, centers, cluster_borders

    @staticmethod
    def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Sample features based on the provided indices.
        """
        new_points = points[idx]
        logger.debug(f"Indexed points with shape: {new_points.shape}")
        return new_points

    def __str__(self) -> str:
        return f"DPCKNNExpand(num_clusters={self.num_clusters}, k={self.k}, clip_model_name='{self.clip_model_name}')"
    
###################################################################################################################################
##################################################################################################################################
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. 
# Do not start writing when i have not given you the context of code

######################################################################################################
class DPCKNNExpand(Operation):
    def __init__(
        self,
        num_clusters: int = 4,
        k: int = 5,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        use_density_for_score: bool = True,
        use_density_minimum_as_border: bool = False,
        reset_seed: bool = False
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.k = k
        self.clip_model_name = clip_model_name
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        if target is None or api is None or graph is None:
            logger.error("Graph, API, and target Node must be provided.")
            return

        # Treat the expansion as an API function regarding seed reset since it is sensitive to non-deterministic behavior
        # This might cause conflicts with summaries that have been extracted before this seed reset was added,
        # that's why it can be controlled by the hyperparameter reset_seed (defaults to False since that was the
        # behavior before the seed reset was added)
        if self.reset_seed:
            logger.debug("Resetting seed as per configuration.")
            api.reset_seed()

        # Get the state of the source node
        source_node_state: NodeState = target.state

        # Get the video clip of the state of the source node
        source_video_clip: VideoClip = source_node_state.video_clip

        video: torch.Tensor = source_video_clip.data  # Assuming video is a tensor of shape (num_frames, C, H, W)
        logger.debug(f"Original video shape: {video.shape}")

        # 1. Get image features using CLIPImageProcessor
        encoder = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = encoder(images=video, return_tensors="pt")
        video_features = inputs["pixel_values"]  # Shape: (num_frames, C', H', W')
        logger.debug(f"video_features.shape: {video_features.shape}")
        # Expected: (num_frames, num_channels, height, width)

        # 2. Transform image features to token features, i.e., flatten to patches of size 1024
        try:
            features = video_features.view(video.size(0), -1, 1024)
            logger.debug(f"Transformed features shape: {features.shape}")
            # Expected: (num_frames, num_patches, 1024)
        except Exception as e:
            logger.error(f"Error reshaping video_features: {e}")
            return

        # 3. Get mean along the second dimension, i.e., mean pooling
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
        logger.debug(f"cls_features.shape after mean pooling: {cls_features.shape}")
        # Shape: (1, num_patches, 1024)

        # Generate the token dictionary
        token_dict = {
            'x': cls_features,  # Shape: (1, num_patches, 1024)
            'token_num': cls_features.size(1),
            'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),  # Shape: (1, num_patches)
            'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),  # Shape: (1, num_patches, 1)
            'mask': None
        }
        logger.debug(f"token_dict['cls_features'].shape: {token_dict['x'].shape}")
        logger.debug(f"token_dict['idx_token'].shape: {token_dict['idx_token'].shape}")
        logger.debug(f"token_dict['agg_weight'].shape: {token_dict['agg_weight'].shape}")

        # 4. Cluster tokens with DPC-KNN algorithm
        cluster_num = min(self.num_clusters, token_dict['x'].size(1))
        k = min(self.k, token_dict['x'].size(1))
        logger.debug(f"Clustering with cluster_num={cluster_num} and k={k} based on video length={token_dict['x'].size(1)}")

        idx_cluster, actual_cluster_num, centers, cluster_borders = self.cluster_dpc_knn(
            token_dict=token_dict,
            cluster_num=cluster_num,
            k=k,
            token_mask=token_dict['mask'],
            use_density_for_score=self.use_density_for_score
        )

        logger.debug(f"Clustering results: actual_cluster_num={actual_cluster_num}, centers={centers}, cluster_borders={cluster_borders}")

        # 5. Sort the cluster centers in ascending order and remove the first frame's center
        if centers.numel() > 1:
            centers_sorted, _ = torch.sort(centers)
            centers_sorted = centers_sorted[1:]  # Remove the first center
            logger.debug(f"Sorted centers (excluding first frame): {centers_sorted}")
        else:
            centers_sorted = centers
            logger.debug("Only one center found; no sorting needed.")

        # 6. Initialize borders based on whether to use density minimum as border
        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video.size(0))
            logger.debug(f"Borders after using density minimum: {borders}")
        else:
            for i in range(len(centers_sorted)):
                if i == len(centers_sorted) - 1:
                    border = video.size(0)
                else:
                    border = int(math.floor((centers_sorted[i + 1] + centers_sorted[i]) / 2))
                borders.append(border)
            logger.debug(f"Borders after calculating mid-points: {borders}")

        # 7. Create new nodes based on the borders
        target_nodes = []
        for i in range(1, len(borders)):
            start_index = borders[i - 1]
            end_index = borders[i]
            logger.debug(f"Creating new node from frame {start_index} to {end_index}")

            # Trim the video data to the split
            split_data = video[start_index:end_index]
            split_sampled_indices = torch.arange(start_index, end_index)

            # Create a new VideoClip
            split_video_clip = VideoClip(data=split_data, sampled_indices=split_sampled_indices)

            # Create a new NodeState
            new_state = NodeState(
                video_clip=split_video_clip,
                task=source_node_state.task,
                lexical_representation=source_node_state.lexical_representation,
                spatial_node_state=SpatialNodeState(
                    video_clip=split_video_clip,
                    task=source_node_state.spatial_node_state.task,
                    use_action_captions=source_node_state.spatial_node_state.use_action_captions,
                    use_object_detections=source_node_state.spatial_node_state.use_object_detections,
                    use_action_captions_summary=source_node_state.spatial_node_state.use_action_captions_summary,
                    use_object_detections_summary=source_node_state.spatial_node_state.use_object_detections_summary,
                    lexical_representation=source_node_state.spatial_node_state.lexical_representation
                ),
                temporal_node_state=TemporalNodeState(
                    video_clip=split_video_clip,
                    task=source_node_state.temporal_node_state.task,
                    use_temporal_grounding_summary=source_node_state.temporal_node_state.use_temporal_grounding_summary,
                    lexical_representation=source_node_state.temporal_node_state.lexical_representation,
                    use_relevance=source_node_state.temporal_node_state.use_relevance,
                    use_salience=source_node_state.temporal_node_state.use_salience,
                    use_foreground=source_node_state.temporal_node_state.use_foreground
                )
            )

            # Create a new Node
            new_node = Node(state=new_state)
            target_nodes.append(new_node)
            logger.debug(f"Created new node with video clip from frame {start_index} to {end_index}")

        # 8. Add new nodes to the graph
        graph.add_nodes(target_nodes)
        logger.debug(f"Added {len(target_nodes)} new nodes to the graph.")

        # 9. Create edges between the source node and the new target nodes
        edges = [Edge(source=target, target_node=new_node) for new_node in target_nodes]
        graph.add_edges(edges)
        logger.debug(f"Added {len(edges)} new edges to the graph.")

    @staticmethod
    def cluster_dpc_knn(
        token_dict: Dict[str, Any],
        cluster_num: int,
        k: int = 5,
        token_mask: Optional[torch.Tensor] = None,
        use_density_for_score: bool = True
    ) -> Any:
        """
        Cluster tokens with DPC-KNN algorithm.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi

        Returns:
            idx_cluster (Tensor[B, N]): Cluster index of each token.
            cluster_num (int): Actual cluster number.
            centers (Tensor[B]): Indices of cluster centers.
            cluster_borders (List[int]): Borders between clusters.
        """
        with torch.no_grad():
            x = token_dict["x"]  # Shape: (B, N, C)
            B, N, C = x.shape
            logger.debug(f"Clustering input x.shape: {x.shape}")

            # Compute pairwise distance matrix
            dist_matrix = torch.cdist(x.float(), x.float()) / math.sqrt(C)
            logger.debug(f"Distance matrix shape: {dist_matrix.shape}")

            if token_mask is not None:
                token_mask = token_mask > 0  # Convert to boolean mask
                # Assign maximum distance to masked-out tokens
                max_dist = dist_matrix.max() + 1
                dist_matrix = dist_matrix * token_mask[:, None, :] + max_dist * (~token_mask)[:, None, :]
                logger.debug("Applied token mask to distance matrix.")

            # Compute local density
            dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)  # Shape: (B, N, k)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # Shape: (B, N)
            # Add a small noise to ensure no tokens have the same density
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            logger.debug(f"Density shape: {density.shape}")

            # Compute delta: minimum distance to higher density
            sorted_density, sorted_indices = torch.sort(density, descending=True)
            delta = torch.zeros_like(density)
            nearest_higher = torch.full_like(sorted_indices, -1)

            for b in range(B):
                for i in range(N):
                    if i == 0:
                        delta[b, sorted_indices[b, i]] = dist_matrix[b, sorted_indices[b, i], sorted_indices[b, i]]
                        nearest_higher[b, sorted_indices[b, i]] = -1
                    else:
                        prev_indices = sorted_indices[b, :i]
                        min_dist, min_idx = torch.min(dist_matrix[b, sorted_indices[b, i], prev_indices], dim=0)
                        delta[b, sorted_indices[b, i]] = min_dist
                        nearest_higher[b, sorted_indices[b, i]] = prev_indices[min_idx]

            logger.debug("Computed delta for each token.")

            # Compute decision score
            if use_density_for_score:
                score = density * delta
            else:
                score = delta
            logger.debug(f"Score shape: {score.shape}")

            # Select cluster centers based on the highest scores
            _, center_indices = torch.topk(score, cluster_num, dim=-1)  # Shape: (B, cluster_num)
            centers = center_indices  # Indices of centers

            # Assign clusters
            idx_cluster = torch.full((B, N), -1, dtype=torch.long, device=x.device)  # Initialize with -1
            for b in range(B):
                for c in range(cluster_num):
                    idx = centers[b, c]
                    idx_cluster[b, idx] = c
                # Assign remaining points to the nearest center
                for n in range(N):
                    if idx_cluster[b, n] == -1:
                        nearest_center = nearest_higher[b, n]
                        if nearest_center == -1:
                            idx_cluster[b, n] = 0  # Assign to first cluster if no higher density
                        else:
                            idx_cluster[b, n] = idx_cluster[b, nearest_center]
            logger.debug("Assigned clusters to all tokens.")

            # Determine cluster borders based on centers
            # For simplicity, assume borders are the sorted frame indices of centers
            centers_sorted, _ = torch.sort(centers, dim=1)
            cluster_borders = centers_sorted.tolist()  # Convert to list for each batch
            logger.debug(f"Cluster borders: {cluster_borders}")

            # For this implementation, assume batch size B=1
            if B != 1:
                logger.warning("Batch size >1 is not fully supported in this implementation.")
            cluster_borders = cluster_borders[0] if B == 1 else cluster_borders

            return idx_cluster, cluster_num, centers, cluster_borders

    @staticmethod
    def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Sample features following the index.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi

        Returns:
            new_points (Tensor): Indexed points data, shape [B, S, C]

        Args:
            points (Tensor): Input points data, shape [B, N, C]
            idx (Tensor): Sample index data, shape [B, S]
        """
        device = points.device
        B = points.shape[0]
        S = idx.shape[1]
        C = points.shape[2]

        # Expand idx for gathering
        idx = idx.unsqueeze(-1).expand(-1, -1, C)  # Shape: [B, S, C]
        new_points = torch.gather(points, 1, idx)  # Shape: [B, S, C]
        logger.debug(f"Indexed points shape: {new_points.shape}")
        return new_points

    def __str__(self) -> str:
        return (
            f"DPCKNNExpand("
            f"num_clusters={self.num_clusters}, "
            f"k={self.k}, "
            f"clip_model_name='{self.clip_model_name}', "
            f"use_density_for_score={self.use_density_for_score}, "
            f"use_density_minimum_as_border={self.use_density_minimum_as_border}, "
            f"reset_seed={self.reset_seed})"
        )
######################################################################################################
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start writing when i still given you my code.
    
######################################################################################################################################3
#######################################################################################################################################3

class DPCKNNExpand(Operation):
    def __init__(
        self,
        num_clusters: int = 4,
        k: int = 5,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        use_density_for_score: bool = True,
        use_density_minimum_as_border: bool = False,
        reset_seed: bool = False
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.k = k
        self.clip_model_name = clip_model_name
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node]
    ) -> None:
        if target is None or api is None or graph is None:
            logger.error("Graph, API, and target Node must be provided.")
            return

        if self.reset_seed:
            logger.debug("Resetting seed.")
            api.reset_seed()

        # Get the state of the source node
        source_node_state: NodeState = target.state

        # Get the video clip of the state of the source node
        source_video_clip: VideoClip = source_node_state.video_clip

        video: torch.Tensor = source_video_clip.data  # Assuming video is a tensor
        logger.debug(f"Original video shape: {video.shape}")

        # Initialize CLIP Image Processor and Encoder
        image_processor = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = image_processor(images=video, return_tensors="pt")
        video_features = inputs['pixel_values']

        logger.debug(f"Video features shape: {video_features.shape}")

        # Transform image features to token features
        try:
            features = video_features.view(video.size(0), -1, 1024)
            logger.debug(f"Transformed features shape: {features.shape}")
        except Exception as e:
            logger.error(f"Error reshaping video_features: {e}")
            return

        # Mean pooling
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
        logger.debug(f"CLS features shape after mean pooling: {cls_features.shape}")

        # Generate token dictionary
        token_dict = {
            'x': cls_features,
            'token_num': cls_features.size(1),
            'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),
            'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),
            'mask': None
        }
        logger.debug(f"token_dict['x'].shape: {token_dict['x'].shape}")
        logger.debug(f"token_dict['idx_token'].shape: {token_dict['idx_token'].shape}")
        logger.debug(f"token_dict['agg_weight'].shape: {token_dict['agg_weight'].shape}")

        # Adjust k and num_clusters based on video length
        video_length = source_video_clip.data.size(0)
        k = min(self.k, video_length)
        num_clusters = min(self.num_clusters, video_length)

        logger.debug(f"Using k={k} and num_clusters={num_clusters} based on video length={video_length}")

        # Cluster tokens using DPC-KNN
        idx_cluster, cluster_num, centers, cluster_borders = self.cluster_dpc_knn(
            token_dict=token_dict,
            cluster_num=num_clusters,
            k=k,
            token_mask=None,
            use_density_for_score=self.use_density_for_score
        )

        logger.debug(f"Clustering results: cluster_num={cluster_num}, centers={centers}, cluster_borders={cluster_borders}")

        # Sort centers and remove the first frame
        centers_sorted, _ = torch.sort(centers)
        centers_sorted = centers_sorted[1:]
        logger.debug(f"Sorted centers (excluding first frame): {centers_sorted}")

        # Initialize borders
        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video.size(0))
            logger.debug(f"Borders after using density minimum: {borders}")
        else:
            for i in range(len(centers_sorted)):
                if i == len(centers_sorted) - 1:
                    border = video.size(0)
                else:
                    border = int(math.floor((centers_sorted[i + 1] + centers_sorted[i]) / 2))
                borders.append(border)
            logger.debug(f"Final borders: {borders}")

        target_nodes = []
        for i in range(len(borders) - 1):
            start_index = borders[i]
            end_index = borders[i + 1]
            logger.debug(f"Creating new node from frame {start_index} to {end_index}")

            # Trim the video data to the split
            split_data = video[start_index:end_index]
            split_sampled_indices = torch.arange(start_index, end_index)

            # Create a new VideoClip
            split_video_clip = VideoClip(data=split_data, sampled_indices=split_sampled_indices)

            # Create a new NodeState
            new_node_state = NodeState(
                video_clip=split_video_clip,
                task=source_node_state.task,
                lexical_representation=source_node_state.lexical_representation,
                spatial_node_state=SpatialNodeState(
                    video_clip=split_video_clip,
                    task=source_node_state.spatial_node_state.task,
                    use_action_captions=source_node_state.spatial_node_state.use_action_captions,
                    use_object_detections=source_node_state.spatial_node_state.use_object_detections,
                    use_action_captions_summary=source_node_state.spatial_node_state.use_action_captions_summary,
                    use_object_detections_summary=source_node_state.spatial_node_state.use_object_detections_summary,
                    lexical_representation=source_node_state.spatial_node_state.lexical_representation
                ),
                temporal_node_state=TemporalNodeState(
                    video_clip=split_video_clip,
                    task=source_node_state.temporal_node_state.task,
                    use_temporal_grounding_summary=source_node_state.temporal_node_state.use_temporal_grounding_summary,
                    lexical_representation=source_node_state.temporal_node_state.lexical_representation,
                    use_relevance=source_node_state.temporal_node_state.use_relevance,
                    use_salience=source_node_state.temporal_node_state.use_salience,
                    use_foreground=source_node_state.temporal_node_state.use_foreground
                )
            )

            # Create a new Node
            new_node = Node(state=new_node_state)
            target_nodes.append(new_node)
            logger.debug(f"Created new node with video clip from frame {start_index} to {end_index}")

        # Add new nodes to the graph
        graph.add_nodes(target_nodes)
        logger.debug(f"Added {len(target_nodes)} new nodes to the graph.")

        # Create edges between source node and target nodes
        edges = [Edge(source=target, target=new_node) for new_node in target_nodes]
        graph.add_edges(edges)
        logger.debug(f"Added {len(edges)} new edges to the graph.")

    @staticmethod
    def cluster_dpc_knn(
        token_dict: Dict[str, Any],
        cluster_num: int,
        k: int = 5,
        token_mask: Optional[torch.Tensor] = None,
        use_density_for_score: bool = True
    ) -> Any:
        """
        Cluster tokens with DPC-KNN algorithm.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi

        Returns:
            idx_cluster (Tensor[B, N]): Cluster index of each token.
            cluster_num (int): Actual cluster number.
            centers (Tensor[B]): Indices of cluster centers.
            cluster_borders (List[int]): Borders between clusters.
        """
        with torch.no_grad():
            x = token_dict["x"]  # Shape: (B, N, C)
            B, N, C = x.shape
            logger.debug(f"Clustering input x.shape: {x.shape}")

            # Compute pairwise distance matrix
            dist_matrix = torch.cdist(x.float(), x.float()) / math.sqrt(C)
            logger.debug(f"Distance matrix shape: {dist_matrix.shape}")

            if token_mask is not None:
                token_mask = token_mask > 0  # Convert to boolean mask
                # Assign maximum distance to masked-out tokens
                max_dist = dist_matrix.max() + 1
                dist_matrix = dist_matrix * token_mask[:, None, :] + max_dist * (~token_mask)[:, None, :]
                logger.debug("Applied token mask to distance matrix.")

            # Compute local density
            dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)  # Shape: (B, N, k)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # Shape: (B, N)
            # Add a small noise to ensure no tokens have the same density
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            logger.debug(f"Density shape: {density.shape}")

            # Compute delta: minimum distance to higher density
            delta = torch.zeros_like(density)
            nearest_higher = torch.full((B, N), -1, dtype=torch.long, device=x.device)

            for b in range(B):
                sorted_density, sorted_indices = torch.sort(density[b], descending=True)
                for i in range(N):
                    if i == 0:
                        delta[b, sorted_indices[i]] = dist_matrix[b, sorted_indices[i], sorted_indices[i]]
                        nearest_higher[b, sorted_indices[i]] = -1
                    else:
                        prev_indices = sorted_indices[:i]
                        min_dist, min_idx = torch.min(dist_matrix[b, sorted_indices[i], prev_indices], dim=0)
                        delta[b, sorted_indices[i]] = min_dist
                        nearest_higher[b, sorted_indices[i]] = prev_indices[min_idx]

            logger.debug("Computed delta for each token.")

            # Compute decision score
            if use_density_for_score:
                score = density * delta
            else:
                score = delta
            logger.debug(f"Score shape: {score.shape}")

            # Select cluster centers based on the highest scores
            _, center_indices = torch.topk(score, cluster_num, dim=-1)  # Shape: (B, cluster_num)
            centers = center_indices  # Indices of centers

            # Assign clusters
            idx_cluster = torch.full((B, N), -1, dtype=torch.long, device=x.device)  # Initialize with -1
            for b in range(B):
                for c in range(cluster_num):
                    idx = centers[b, c]
                    idx_cluster[b, idx] = c
                # Assign remaining points to the nearest center
                for n in range(N):
                    if idx_cluster[b, n] == -1:
                        nearest_center = nearest_higher[b, n]
                        if nearest_center == -1:
                            idx_cluster[b, n] = 0  # Assign to first cluster if no higher density
                        else:
                            idx_cluster[b, n] = idx_cluster[b, nearest_center]
            logger.debug("Assigned clusters to all tokens.")

            # Determine cluster borders based on centers
            # For simplicity, assume borders are the sorted frame indices of centers
            centers_sorted, _ = torch.sort(centers, dim=1)
            cluster_borders = centers_sorted.tolist()  # Convert to list for each batch
            logger.debug(f"Cluster borders: {cluster_borders}")

            # For this implementation, assume batch size B=1
            if B != 1:
                logger.warning("Batch size >1 is not fully supported in this implementation.")
            cluster_borders = cluster_borders[0] if B == 1 else cluster_borders

            return idx_cluster, cluster_num, centers, cluster_borders

    @staticmethod
    def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Sample features based on the provided indices.

        Returns:
            new_points (Tensor): Indexed points data, shape [B, S, C]

        Args:
            points (Tensor): Input points data, shape [B, N, C]
            idx (Tensor): Sample index data, shape [B, S]
        """
        B, S = idx.shape
        C = points.shape[2]
        device = points.device

        # Expand idx for gathering
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # Shape: [B, S, C]
        new_points = torch.gather(points, 1, idx_expanded)  # Shape: [B, S, C]
        logger.debug(f"Indexed points shape: {new_points.shape}")
        return new_points

    def __str__(self) -> str:
        return (
            f"DPCKNNExpand("
            f"num_clusters={self.num_clusters}, "
            f"k={self.k}, "
            f"clip_model_name='{self.clip_model_name}', "
            f"use_density_for_score={self.use_density_for_score}, "
            f"use_density_minimum_as_border={self.use_density_minimum_as_border}, "
            f"reset_seed={self.reset_seed})"
        )
