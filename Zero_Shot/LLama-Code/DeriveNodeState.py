#####################################################3
# Instruction
#####################################################33

class DeriveNodeStates(Operation):
    def __init__(self,
                 derive_action_captions: bool,
                 derive_action_captions_summary: bool,
                 total_num_words_action_captions_summary: int,
                 min_num_words_action_captions_summary: int,
                 derive_object_detections: bool,
                 derive_object_detections_summary: bool,
                 total_num_words_object_detections_summary: int,
                 min_num_words_object_detections_summary: int,
                 derive_temporal_grounding: bool,
                 derive_temporal_grounding_summary: bool,
                 total_num_words_temporal_grounding_summary: int,
                 min_num_words_temporal_grounding_summary: int,
                 normalization_video_length: int = 180):
        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node] = None) -> None:
        root_state = graph.root.state
        if not root_state.derived:
            logging.error("Root node state has not been derived yet.")
            return

        derivable_nodes = graph.get_derivable_nodes()
        for derivable_node in derivable_nodes:
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices <sup> </sup>
            total_end_frame_index = derivable_node.state.video_clip.sampled_indices[-1]

            start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
            end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)

            len_clip_sec = (derivable_node.state.video_clip.sampled_fps * (total_end_frame_index - total_start_frame_index + 1)) / 1000
            len_video_sec = root_state.video_clip.sampled_fps * len(root_state.video_clip.sampled_indices) / 1000

            logging.debug(f"Derivable node: {derivable_node}, start index: {start_list_index}, end index: {end_list_index}")
            logging.debug(f"Clip length: {len_clip_sec} sec, Video length: {len_video_sec} sec")

            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions
                logging.info(f"Derived action captions for node: {derivable_node}")
                logging.debug(f"Action captions: {clip_node_action_captions}")

            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(len_clip_sec, len_video_sec, self.total_num_words_action_captions_summary, self.min_num_words_action_captions_summary)
                logging.debug(f"Number of words for action captions summary: {num_words}")
                clip_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                logging.info(f"Derived action captions summary for node: {derivable_node}")
                logging.debug(f"Action captions summary: {clip_summary}")

            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections
                logging.info(f"Derived object detections for node: {derivable_node}")
                logging.debug(f"Object detections: {clip_node_object_detections}")

            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(len_clip_sec, len_video_sec, self.total_num_words_object_detections_summary, self.min_num_words_object_detections_summary)
                logging.debug(f"Number of words for object detections summary: {num_words}")
                object_detection_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                logging.info(f"Derived object detections summary for node: {derivable_node}")
                logging.debug(f"Object detections summary: {object_detection_summary}")

            if self.derive_temporal_grounding:
                derivable_node.state.temporal_node_state.foreground_indicators = root_state.temporal_node_state.foreground_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.relevance_indicators = root_state.temporal_node_state.relevance_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.salience_indicators = root_state.temporal_node_state.salience_indicators[start_list_index:end_list_index + 1]
                logging.info(f"Derived temporal grounding for node: {derivable_node}")
                logging.debug(f"Temporal grounding: {derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(len_clip_sec, len_video_sec, self.total_num_words_temporal_grounding_summary, self.min_num_words_temporal_grounding_summary)
                logging.debug(f"Number of words for temporal grounding summary: {num_words}")
                temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )
                logging.info(f"Derived temporal grounding summary for node: {derivable_node}")
                logging.debug(f"Temporal grounding summary: {temporal_grounding_summary}")

            derivable_node.state.derived = True
            logging.info(f"Derived node state: {derivable_node.state}")

        logging.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(self, len_clip_sec: float, len_video_sec: float, total_num_words_summary: int, min_num_words_summary: int) -> int:
        if self.normalization_video_length == -1:
            total_num_words_summary = total_num_words_summary
        else:
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
            num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
            num_words_mul_ten = num_words_exact - (num_words_exact % 10)
            return max(num_words_mul_ten, min_num_words_summary)
        
################################################################################################################################################################
class DeriveNodeStates(Operation):
    def __init__(
            self,
            derive_action_captions: bool,
            derive_action_captions_summary: bool,
            total_num_words_action_captions_summary: int,
            min_num_words_action_captions_summary: int,
            derive_object_detections: bool,
            derive_object_detections_summary: bool,
            total_num_words_object_detections_summary: int,
            min_num_words_object_detections_summary: int,
            derive_temporal_grounding: bool,
            derive_temporal_grounding_summary: bool,
            total_num_words_temporal_grounding_summary: int,
            min_num_words_temporal_grounding_summary: int,
            normalization_video_length: int = 180
    ):
        super().__init__()

        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional[Graph], api: Optional[API], target=Optional[Node]) -> None:
        # get the root node state from the graph, because we can inherit the perceptive information
        root_state = graph.root.state

        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        derivable_nodes = graph.get_derivable_nodes()
        for derivable_node in derivable_nodes:

            # find start and end indices of the video clip
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices <sup> </sup>
            total_end_frame_index = derivable_node.state.video_clip.sampled_indices[-1]
            logger.debug(f"Total start frame index: {total_start_frame_index}")
            logger.debug(f"Total end frame index: {total_end_frame_index}")

            # get the list index in the sampled indices of the root node state video
            start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
            end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
            logger.debug(f"Start list index: {start_list_index}")
            logger.debug(f"End list index: {end_list_index}")

            # get the length of the clip and the whole video in seconds respectively
            len_clip_sec = len(derivable_node.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logger.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logger.debug(f"Length of the video in seconds: {len_video_sec}")

            if self.derive_action_captions:
                # inherit the action captions from the root node
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                                            start_list_index:end_list_index + 1]

                # set the action captions of the derivable node
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions

                logger.info(f"Inherited action captions from root node for node {derivable_node}.")
                logger.debug(f"Action captions for node {derivable_node}: {clip_node_action_captions}")

            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec,
                    len_video_sec,
                    self.total_num_words_action_captions_summary,
                    self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action captions summary: {num_words}")

                summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_capt
                    
################################################################################################################################################
class DeriveNodeStates(Operation):
    def __init__(
            self,
            derive_action_captions: bool,
            derive_action_captions_summary: bool,
            total_num_words_action_captions_summary: int,
            min_num_words_action_captions_summary: int,
            derive_object_detections: bool,
            derive_object_detections_summary: bool,
            total_num_words_object_detections_summary: int,
            min_num_words_object_detections_summary: int,
            derive_temporal_grounding: bool,
            derive_temporal_grounding_summary: bool,
            total_num_words_temporal_grounding_summary: int,
            min_num_words_temporal_grounding_summary: int,
            normalization_video_length: int = 180
    ):
        super().__init__()

        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node] = None) -> None:
        # get the root node state from the graph, because we can inherit the perceptive information
        root_state = graph.root.state

        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logging.error(err_msg)
            raise ValueError(err_msg)

        derivable_nodes = graph.get_derivable_nodes()
        for derivable_node in derivable_nodes:

            # find start and end indices of the video clip
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices <sup> </sup>
            total_end_frame_index = derivable_node.state.video_clip.sampled_indices[-1]
            logging.debug(f"Total start frame index: {total_start_frame_index}")
            logging.debug(f"Total end frame index: {total_end_frame_index}")

            # get the list index in the sampled indices of the root node state video
            start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
            end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
            logging.debug(f"Start list index: {start_list_index}")
            logging.debug(f"End list index: {end_list_index}")

            # get the length of the clip and the whole video in seconds respectively
            len_clip_sec = len(derivable_node.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logging.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logging.debug(f"Length of the video in seconds: {len_video_sec}")

            if self.derive_action_captions:
                # inherit the action captions from the root node
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                                            start_list_index:end_list_index + 1]

                # set the action captions of the derivable node
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions

                logging.info(f"Inherited action captions from root node for node {derivable_node}.")
                logging.debug(f"Action Captions:\n{derivable_node.state.spatial_node_state.action_captions}")

            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
                logging.debug(f"Number of words for action caption summary: {num_words}")

                # use action caption summary API function to summarize all action captions to get a clip summary
                derivable_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,  # will be used if it is given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logging.info(f"Derived action caption summary for node {derivable_node}.")
                logging.debug(
                    f"Action Caption Summary:\n{derivable_node.state.spatial_node_state.action_captions_summary}")

            if self.derive_object_detections:
                # inherit the object detections from the root node
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                                              start_list_index:end_list_index + 1]

                # set the object detections of the derivable node
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections

                logging.info(f"Inherited object detections from root node for node {derivable_node}.")
                logging.debug(f"Object Detections:\n"
                             f"{derivable_node.state.spatial_node_state.object_detections}")

            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
                logging.debug(f"Number of words for object detections summary: {num_words}")

                # use object detection summary API function to summarize all object detections to get a universal video summary
                derivable_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,  # will be used if it is given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logging.info(f"Derived object detection summary for node {derivable_node}.")
                logging.debug(f"Object Detection Summary:\n"
                             f"{derivable_node.state.spatial_node_state.object_detections_summary}")

            if self.derive_temporal_grounding:
                # inherit the temporal grounding from the root node
                derivable_node.state.temporal_node_state.foreground_indicators = root_state.temporal_node_state.foreground_indicators[
                                                                                 start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.relevance_indicators = root_state.temporal_node_state.relevance_indicators[
                                                                                start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.salience_indicators = root_state.temporal_node_state.salience_indicators[
                                                                               start_list_index:end_list_index + 1]

                logging.info(f"Inherited temporal grounding from root node for node {derivable_node}.")
                logging.debug(f"Temporal Grounding:\n"
                             f"{derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
                logging.debug(f"Number of words for temporal grounding summary: {num_words}")

                # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
                derivable_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,  # will be used if it is given in the prompt template
                    options=derivable_node.state.task.options,  # will be used if they are given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logging.info(f"Derived temporal grounding summary for node {derivable_node}.")
                logging.debug(f"Temporal Grounding Summary:\n"
                             f"{derivable_node.state.temporal_node_state.temporal_grounding_summary}")

            derivable_node.state.derived
 ###############################################################################################################