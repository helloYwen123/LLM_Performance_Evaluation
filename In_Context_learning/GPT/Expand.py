#############################################################3
class Expand(Operation):
    def __init__(
        self,
        num_splits: int = 4
    ):
        super().__init__()
        self.num_splits = num_splits

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        if target is None:
            logger.error("Target node is None.")
            return
        
        # Get the source video clip from the target node's state
        source_node_state = target.state
        source_video_clip = source_node_state.video_clip
        video_length = len(source_video_clip)
        
        logger.debug(f"Source video length: {video_length}")

        if video_length == 0:
            logger.warning("Source video clip is empty.")
            return

        # Generate indices and split them into chunks
        indices = torch.tensor(list(range(video_length)))
        splits = torch.chunk(indices, self.num_splits) if video_length >= self.num_splits else torch.chunk(indices, video_length)
        
        logger.debug(f"Number of splits: {len(splits)}")

        target_nodes = []
        
        for split in splits:
            if split.numel() == 0:
                continue  # Skip empty splits

            start_list_index = split[0].item()
            end_list_index = split[-1].item()
            
            logger.debug(f"Processing split from index {start_list_index} to {end_list_index}")

            # Trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

            # Create a new VideoClip instance
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            # Create a new Node with the new VideoClip
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
            logger.debug(f"Created new node for split: {new_node}")

        if not target_nodes:
            logger.warning("No target nodes were created from the splits.")
            return

        # Create edges from the source node to each target node
        edges = [Edge(source=target, target_node=new_node) for new_node in target_nodes]
        
        # Add the new nodes and edges to the graph
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

        logger.info(f"Executed Expand operation: Split video into {len(target_nodes)} segments.")

    def __str__(self):
        return f"Expand(num_splits={self.num_splits})"
    ###########################################################################################
    
    class Expand(Operation):
    def __init__(self, num_splits: int = 4):
        """
        Initialize the Expand operation.

        Args:
            num_splits (int, optional): Number of splits to divide the video into. Defaults to 4.
        """
        super().__init__()
        self.num_splits = num_splits

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        """
        Execute the Expand operation to split the video clip into multiple segments.

        Args:
            graph (Optional[Graph]): The graph to which new nodes and edges will be added.
            api (Optional[API]): The API instance for any required operations.
            target (Optional[Node]): The target node containing the video clip to be split.
        """
        if target is None:
            logger.error("Target node is None.")
            return

        # Get the video clip from the target node's state
        source_node_state = target.state
        source_video_clip = source_node_state.video_clip

        video_length = len(source_video_clip)
        logger.debug(f"Source video length: {video_length}")

        if video_length == 0:
            logger.warning("Source video clip is empty.")
            return

        # Generate indices and split them into chunks
        indices = torch.tensor(list(range(video_length)))
        actual_num_splits = self.num_splits if video_length >= self.num_splits else video_length
        splits = torch.chunk(indices, actual_num_splits)
        logger.debug(f"Number of splits: {len(splits)}")

        target_nodes = []

        for split in splits:
            if split.numel() == 0:
                logger.debug("Encountered an empty split; skipping.")
                continue  # Skip empty splits

            # Get the start and end index of the split
            start_list_index = split[0].item()
            end_list_index = split[-1].item()
            logger.debug(f"Processing split from index {start_list_index} to {end_list_index}")

            # Trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]
            logger.debug(f"Split data shape: {split_data.shape}")

            # Trim the sampled indices to the split
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]
            logger.debug(f"Split sampled indices: {split_sampled_indices}")

            # Create a new VideoClip instance
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )
            logger.debug(f"Created split VideoClip: {split_video_clip}")

            # Create a new Node with the new VideoClip
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
                        use_foreground=source_node_state.temporal_node_state.use_foreground,
                        use_salience=source_node_state.temporal_node_state.use_salience
                    )
                )
            )
            logger.debug(f"Created new node for split: {new_node}")

            target_nodes.append(new_node)

        if not target_nodes:
            logger.warning("No target nodes were created from the splits.")
            return

        # Create edges from the source node to each target node
        edges = [Edge(source=target, target_node=new_node) for new_node in target_nodes]
        logger.debug(f"Created {len(edges)} edges.")

        # Add the new nodes and edges to the graph
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)
        logger.debug("Added new nodes and edges to the graph.")

        logger.info(f"Executed structure expansion operation: Expand with {len(target_nodes)} splits.")

    def __str__(self):
        """
        Return a string representation of the Expand instance.

        Returns:
            str: String representation.
        """
        return f"Expand(num_splits={self.num_splits})"
    ##############################################################################################
   class Expand(Operation):
    def __init__(self, num_splits: int = 4):
        super().__init__()
        self.num_splits = num_splits

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        source_video_clip = target.state.video_clip
        indices = torch.tensor(list(range(len(source_video_clip))))
        splits = torch.chunk(indices, self.num_splits)

        target_nodes = []
        for split in splits:
            start_index = split[0].item()
            end_index = split[-1].item()
            split_data = source_video_clip.data[start_index:end_index + 1]
            split_sampled_indices = source_video_clip.sampled_indices[start_index:end_index + 1]

            new_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            new_node = Node(
                state=NodeState(
                    video_clip=new_video_clip,
                    task=target.state.task,
                    lexical_representation=target.state.lexical_representation,
                    spatial_node_state=target.state.spatial_node_state,
                    temporal_node_state=target.state.temporal_node_state
                )
            )

            target_nodes.append(new_node)

        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes]
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

        logger.info(f"Executed expansion operation: Expand(num_splits={self.num_splits})")

    def __str__(self):
        return f"Expand(num_splits={self.num_splits})"
		