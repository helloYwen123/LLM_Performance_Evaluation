class DeriveRootNodeState(Operation):
    def __init__(
            self,
            derive_action_captions: bool,
            derive_action_captions_summary: bool,
            num_words_action_captions_summary: int,
            min_num_words_action_captions_summary: int,
            derive_object_detections: bool,
            derive_object_detections_summary: bool,
            num_words_object_detections_summary: int,
            min_num_words_object_detections_summary: int,
            derive_temporal_grounding: bool,
            derive_temporal_grounding_summary: bool,
            num_words_temporal_grounding_summary: int,
            min_num_words_temporal_grounding_summary: int,
            normalization_video_length: int = 180
    ):
        super().__init__()

        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.num_words_action_captions_summary = num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.num_words_object_detections_summary = num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.num_words_temporal_grounding_summary = num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional[Graph], api: Optional[API], target=Optional[Node]) -> None:
        root_node = graph.root
        len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

        if self.derive_action_captions:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=root_node.state.spatial_node_state.action_captions,
                object_detections=[],
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,  # will be used if it is given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived action caption summary for root node {root_node}.")
            logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

        if self.derive_object_detections:
            # use object detection API function to get object detections from the video clip
            root_node.state.spatial_node_state.object_detections = api.get_unspecific_objects_from_video_clip(
                video_clip=root_node.state.video_clip)

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n"
                         f"{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            # use object detection summary API function to summarize all object detections to get a universal video summary
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=root_node.state.spatial_node_state.get_textual_object_list(),
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,  # will be used if it is given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived object detection summary for root node {root_node}.")
            logger.debug(f"Object Detection Summary:\n"
                         f"{root_node.state.spatial_node_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            # use temporal grounding API function to get temporal grounding from the video clip and the question
            temporal_grounding = api.get_temporal_grounding_from_video_clip_and_text(
                video_clip=root_node.state.video_clip,
                text=root_node.state.task.question
            )

            root_node.state.temporal_node_state.foreground_indicators = temporal_grounding["foreground_indicators"]
            root_node.state.temporal_node_state.relevance_indicators = temporal_grounding["relevance_indicators"]
            root_node.state.temporal_node_state.salience_indicators = temporal_grounding["salience_indicators"]

            logger.info(f"Derived temporal grounding for root node {root_node}.")
            logger.debug(f"Temporal Grounding:\n"
                         f"{root_node.state.temporal_node_state.get_textual_temporal_grounding()}")

        if self.derive_temporal_grounding_summary:
            # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
            root_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=[],
                temporal_grounding=[root_node.state.temporal_node_state.get_textual_temporal_grounding()],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,  # will be used if it is given in the prompt template
                options=root_node.state.task.options,  # will be used if they are given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived temporal grounding summary for root node {root_node}.")
            logger.debug(f"Temporal Grounding Summary:\n"
                         f"{root_node.state.temporal_node_state.temporal_grounding_summary}")

        root_node.state.derived = True

        logger.info(f"Derived universal node state for root node {root_node}.")
        logger.debug(f"The following universal node state has been derived:\n{root_node.state}")
        logger.info(f"Executed root state derivation operation: DeriveUniversalState")

    def get_num_words_for_summary(
            self,
            len_video_sec,
            total_num_words_summary,
            min_num_words_summary
    ):
        # do not normalize the number of words for the summary if the normalization video length is -1
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
        else:
            # calculate the number of words for the summary (proportionate to the video length in seconds)
            # calculate the word contingent for the whole video
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
        # make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(whole_video_word_contingent / 10))) * 10
        # clip the number of words to the minimum number of words
        num_words = max(num_words_mul_ten, min_num_words_summary)
        return num_words