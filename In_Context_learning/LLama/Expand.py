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
            start_list_index = split[0].item()
            end_list_index = split[-1].item()
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )
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
                        lexical_representation=target.state.spatial_node_state.lexical_representation
                    ),
                    temporal_node_state=TemporalNodeState(
                        video_clip=split_video_clip,
                        task=target.state.temporal_node_state.task,
                        use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
                        lexical_representation=target.state.temporal_node_state.lexical_representation,
                        use_relevance=target.state.temporal_node_state.use_relevance,
                        use_salience=target.state.temporal_node_state.use_salience,
                        use_foreground=target.state.temporal_node_state.use_foreground
                    )
                )
            )
            target_nodes.append(new_node)

        edges = [Edge(source=target, target_node=target_node) for target_node in target_nodes]
        graph.add_nodes(target_nodes)
        graph.add_edges(edges)
        logger.info("Executed structure expansion operation: Expand")

    def __str__(self):
        return f"Expand(num_splits={self.num_splits})"
    #################################################################################3
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
                        lexical_representation=target.state.spatial_node_state.lexical_representation
                    ),
                    temporal_node_state=TemporalNodeState(
                        video_clip=split_video_clip,
                        task=target.state.temporal_node_state.task,
                        use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
                        lexical_representation=target.state.temporal_node_state.lexical_representation,
                        use_relevance=target.state.temporal_node_state.use_relevance,
                        use_foreground=target.state.temporal_node_state.use_foreground,
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
    ###########################################################################3
    