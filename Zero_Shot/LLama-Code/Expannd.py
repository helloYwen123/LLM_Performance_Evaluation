###############################
######Instruct#################
###############################
class Expand(Operation):
    def __init__(self, num_splits: int = 4):
        self.num_splits = num_splits

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        source_video_clip = target.state.video_clip
        indices = torch.tensor(list(range(len(source_video_clip))))
        splits = torch.chunk(indices, self.num_splits)

        target_nodes = []
        for split in splits:
            start_list_index, end_list_index = split[0], split[-1]
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

            new_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            new_node = Node(
                video_clip=new_video_clip,
                task=target.task,
                lexical_representation=target.lexical_representation,
                spatial_node_state=target.spatial_node_state,
                temporal_node_state=target.temporal_node_state
            )

            target_nodes.append(new_node)

        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes]
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)
        logger.info(f"Added {len(target_nodes)} new nodes and {len(edges)} new edges to the graph")

    def __str__(self):
        return str(self.num_splits)
############################################################################

class Expand(Operation):
    def __init__(self, num_splits: int = 4):
        """
        Initialize the Expand operation.

        Args:
            num_splits (int): Number of splits to divide the video clip into.
        """
        super().__init__()
        self.num_splits = num_splits

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        """
        Execute the expand operation on the target node.

        Args:
            graph (Optional[Graph]): The graph to which nodes and edges will be added.
            api (Optional[API]): API instance (if needed for extended functionality).
            target (Optional[Node]): The target node to expand.

        Returns:
            None
        """
        if target is None:
            logger.error("Target node is None. Cannot execute Expand operation.")
            return

        if target.state.video_clip is None:
            logger.error("Target node does not have a video clip. Cannot execute Expand operation.")
            return

        source_video_clip = target.state.video_clip
        total_frames = len(source_video_clip.data)
        logger.info(f"Total frames in source video clip: {total_frames}")

        # Generate indices and split them
        indices = torch.tensor(list(range(total_frames)))
        splits = torch.chunk(indices, self.num_splits)

        target_nodes = []
        for split in splits:
            if split.numel() == 0:
                logger.warning("Encountered an empty split. Skipping.")
                continue

            # Get the start and end index of the split
            start_list_index = split[0].item()
            end_list_index = split[-1].item()
            logger.debug(f"Processing split from index {start_list_index} to {end_list_index}")

            # Trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]

            # Trim the sampled indices to the split
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
            logger.debug("Created new VideoClip for the split.")

            # Create a new SpatialNodeState instance
            if target.state.spatial_node_state:
                new_spatial_node_state = SpatialNodeState(
                    video_clip=split_video_clip,
                    task=target.state.spatial_node_state.task,
                    use_action_captions=target.state.spatial_node_state.use_action_captions,
                    use_object_detections=target.state.spatial_node_state.use_object_detections,
                    use_action_captions_summary=target.state.spatial_node_state.use_action_captions_summary,
                    use_object_detections_summary=target.state.spatial_node_state.use_object_detections_summary,
                    lexical_representation=target.state.spatial_node_state.lexical_representation
                )
                logger.debug("Created new SpatialNodeState for the split.")
            else:
                new_spatial_node_state = None
                logger.warning("Target node does not have a SpatialNodeState. Skipping SpatialNodeState creation.")

            # Create a new TemporalNodeState instance
            if target.state.temporal_node_state:
                new_temporal_node_state = TemporalNodeState(
                    video_clip=split_video_clip,
                    task=target.state.temporal_node_state.task,
                    use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
                    lexical_representation=target.state.temporal_node_state.lexical_representation,
                    use_relevance=target.state.temporal_node_state.use_relevance,
                    use_foreground=target.state.temporal_node_state.use_foreground,
                    use_salience=target.state.temporal_node_state.use_salience
                )
                logger.debug("Created new TemporalNodeState for the split.")
            else:
                new_temporal_node_state = None
                logger.warning("Target node does not have a TemporalNodeState. Skipping TemporalNodeState creation.")

            # Create a new NodeState instance
            new_node_state = NodeState(
                video_clip=split_video_clip,
                task=target.state.task,
                lexical_representation=target.state.lexical_representation,
                spatial_node_state=new_spatial_node_state,
                temporal_node_state=new_temporal_node_state
            )
            logger.debug("Created new NodeState for the split.")

            # Create a new Node instance
            new_node = Node(state=new_node_state)
            logger.debug("Created new Node for the split.")

            # Append the new node to the list of target nodes
            target_nodes.append(new_node)

        if not target_nodes:
            logger.warning("No target nodes created from the splits.")
            return

        # Create edges from the original target node to each new node
        edges = [Edge(source=target, target_node=new_node) for new_node in target_nodes]
        logger.debug(f"Created {len(edges)} edges for the new nodes.")

        # Add new nodes and edges to the graph
        if graph:
            graph.add_nodes(target_nodes)
            graph.add_edges(edges)
            logger.info(f"Expanded target node into {len(target_nodes)} new nodes with {len(edges)} edges.")
        else:
            logger.error("Graph is None. Cannot add nodes and edges.")

    def __str__(self):
        return f"Expand(num_splits={self.num_splits})"
    ###############################################################################################3
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
            # get the start and end index of the split
            start_list_index = split[0].item()
            end_list_index = split[-1].item()

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
                    task=target.state.task,
                    lexical_representation=target.state.lexical_representation,
                    spatial_node_state=SpatialNodeState(
                        video_clip=split_video_clip,
                        task=target.state.spatial_node_state.task,
                        use_action_captions=target.state.spatial_node_state.use_action_captions,
                        use_object_detections=target.state.spatial_node_state.use_object_detections,
                        use_action_captions_summary=target.state.spatial_node_state.use_action_captions_summary,
                        use_object_detections_summary=target.state.spatial_node_state.use_object_detections_summary,
                        lexical_representation=target.state.lexical_representation
                    ),
                    temporal_node_state=TemporalNodeState(
                        video_clip=split_video_clip,
                        task=target.state.temporal_node_state.task,
                        use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
                        use_salience=target.state.temporal_node_state.use_salience
                    )
                )
            )

            target_nodes.append(new_node)

        # apply the expansion to the graph
        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes]

        graph.add_nodes(target_nodes)
        graph.add_edges(edges)

        logger.info(f"Executed structure expansion operation: Expand")

    def __str__(self):
        return f"Expand(num_splits={self.num_splits})"