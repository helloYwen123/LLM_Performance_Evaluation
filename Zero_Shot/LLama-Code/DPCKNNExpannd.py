# instruction zero-shot
import torch
from torch import nn
from transformers import CLIPImageProcessor
from typing import Optional
from your_module import Graph, API, Node, NodeState, SpatialNodeState, TemporalNodeState, VideoClip, Edge

class DPCKNNExpand(Operation):
    def __init__(self, num_clusters: int = 4, k: int = 5, clip_model_name: str = "openai/clip-vit-large-patch14", 
                 use_density_for_score: bool = True, use_density_minimum_as_border: bool = False, reset_seed: bool = False):
        self.num_clusters = num_clusters
        self.clip_model_name = clip_model_name
        self.k = k
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]):
        if self.reset_seed:
            api.reset_seed()
        source_node_state = target.state
        source_video_clip = source_node_state.video_clip

        image_encoder = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = image_encoder(images=source_video_clip, return_tensors="pt")
        video_features = inputs["pixel_values"]

        # transform image feature to token feature and flatten to patches of size 1024
        features = video_features.view(video_features.size(0), -1, 1024)

        # mean pooling for features
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()

        logger.debug(f"Processed features shape: {cls_features.shape}")

        token_dict = {
            'x': cls_features,
            'token_num': cls_features.size(1),
            'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),
            'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),
           'mask': None
        }

        k = min(self.k, len(source_video_clip))
        num_clusters = min(self.num_clusters, len(source_video_clip))

        idx_cluster, cluster_num, centers, cluster_borders = self.cluster_dpc_knn(token_dict, num_clusters, k, None, self.use_density_for_score)

        # sort cluster centers in ascending order and remove the first frame
        centers = sorted(centers)[1:]

        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video_features.size(0))
        else:
            for i in range(len(centers)):
                if i == len(centers) - 1:
                    borders.append(video_features.size(0))
                else:
                    border = (centers[i] + centers[i + 1]) // 2
                    borders.append(border)

        target_nodes = []
        for i in range(len(borders) - 1):
            split_data = source_video_clip[borders[i]:borders[i + 1]]
            split_sampled_indices = source_node_state.sampled_indices[borders[i]:borders[i + 1]]
            split_video_clip = VideoClip(split_data, split_sampled_indices)
            new_node = Node(NodeState(split_video_clip, SpatialNodeState(), TemporalNodeState()))
            target_nodes.append(new_node)

        # expand edges and graph
        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes for source_node in [target]]
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

    @staticmethod
    def cluster_dpc_knn(token_dict, cluster_num, k, token_mask=None, use_density_for_score=True):
        # implementation of DPC-KNN clustering algorithm
        #...
        return idx_cluster, index_down, borders

    @staticmethod
    def index_points(points, idx):
        # implementation of indexing points
        #...
        return new_points

    def __str__(self):
        return f"DPCKNNExpand(num_clusters={self.num_clusters}, k={self.k}, clip_model_name={self.clip_model_name})"
    ####################################################################################################################
    #completion
    ###################################################################################################################
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

        # cluster tokens with DPC-KNN algorithm
        k = min(self.k, len(source_video_clip))
        num_clusters = min(self.num_clusters, len(source_video_clip))
        idx_cluster, cluster_num, centers, cluster_borders = self.cluster_dpc_knn(token_dict, num_clusters, k, None, self.use_density_for_score)

        # get the cluster centers and sort them in ascending order and then remove the first frame
        centers = sorted(centers)[1:]

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

            split_data = source_video_clip.data[start_list_index:end_list_index]
            split_sampled_indices = source_node_state.sampled_indices[start_list_index:end_list_index]
            split_video_clip = VideoClip(split_data, split_sampled_indices)

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

        # apply the expansion to the Edges and graph
        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes for source_node in [target]]
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

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
            density = (-(dist_nearest ** 2).exp())
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            logger.debug(f"density: {density.shape}")

            # compute cluster centers
            idx_cluster, index_down = torch.topk(density, k=cluster_num, dim=-1, largest=True)
            logger.debug(f"idx_cluster: {idx_cluster.shape}")

            # compute cluster borders
            cluster_centers = x[:, index_down[0]]
            cluster_centers = cluster_centers.unsqueeze(1).repeat(1, N, 1)
            dist_centers = torch.cdist(x, cluster_centers, p=2)
            cluster_borders = torch.topk(dist_centers, k=1, dim=-1, largest=False)[1]
            logger.debug(f"cluster_borders: {cluster_borders.shape}")

            return idx_cluster, index_down, cluster_b
#######################################################################################################################
######################################################################################################################

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
        graph.add_nodes(target_nodes)  # Corrected the function call
        logger.debug(f"Added {len(target_nodes)} new nodes to the graph.")

        # Create edges between source node and target nodes
        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes for source_node in [target]]
        graph.add_edges(edges)  # Corrected the function call
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