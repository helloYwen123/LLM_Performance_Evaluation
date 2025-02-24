# #######################################################
# #######################################################
# Based on Description to generate code
# #######################################################
# #######################################################
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
        normalization_video_length: int = 180,
    ):
        super().__init__()
        self.derive_action_captions = derive_action_captions
        self.derive_action_captions_summary = derive_action_captions_summary
        self.num_words_action_captions_summary = num_words_action_captions_summary
        self.min_num_words_action_captions_summary = (
            min_num_words_action_captions_summary or 5
        )  # Default value added here since `or` operator can't handle default values directly within initialization logic
        self.derive_object_detections = derive_object_detections
        self.derive_object_detections_summary = derive_object_detections_summary
        self.num_words_object_detections_summary = num_words_object_detections_summary
        self.min_num_words_object_detections_summary = (
            min_num_words_object_detections_summary or 5
        )  # Default value added similarly
        self.derive_temporal_grounding = derive_temporal_grounding
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.num_words_temporal_grounding_summary = num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = (
            min_num_words_temporal_grounding_summary or 5
        )  # Default value added similarly
        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]):
        assert isinstance(graph, Graph)
        assert isinstance(api, API)
        assert isinstance(target, Node)

        root_node = graph.root_node
        video_clip = root_node.state.video_clip
        fps = video_clip.sampled_fps
        duration_seconds = (len(video_clip) / fps).round().astype(int)

        if self.derive_action_captions:
            action_captions = api.get_action_captions(root_node.state.video_clip)
            root_node.state.spatial_node_state.action_captions = action_captions
            print("Action Captions derived.")
            logging.debug(f"Action Captions: {action_captions}")

        if self.derive_action_captions_summary:
            action_captions_summary = api.summary_from_noisy_perceptual_data(
                action_captions=root_node.state.spatial_node_state.action_captions,
                question="",
                words=self.num_words_action_captions_summary,
            )
            root_node.state.spatial_node_state.action_captions_summary = action_captions_summary
            print("Action Captions Summary derived.")
            logging.debug(f"Action Captions Summary: {action_captions_summary}")

        if self.derive_object_detections:
            _, object_detections = api.get_unspecific_objects_from_video_clip(
                action_captions=root_node.state.spatial_node_state.action_captions,
                object_detections=None,
                temporal_grounding=None,
                interleaved_action_captions_and_object_detections=False,
                video_clip=root_node.state.video_clip,
                question="",
                words=self.num_words_object_detections_summary,
            )
            root_node.state.spatial_node_state.object_detections = object_detections
            print("Object Detections derived.")
            logging.debug(f"Object Detections: {object_detections}")

        if self.derive_temporal_grounding:
            foreground_indicator, relevance_indicator, salience_indicator = api.get_temporal_grounding_from_video_clip_and_text(
                video_clip=root_node.state.video_clip,
                question=""
            )
            root_node.state.spatial_node_state.temporal_grounding = {
                "foreground": foreground_indicator,
                "relevance": relevance_indicator,
                "salience": salience_indicator
            }
            print("Temporal Grounding derived.")
            logging.debug(f"Foreground Indicator: {foreground_indicator}")
            logging.debug(f"Relevance Indicator: {relevance_indicator}")
            logging.debug(f"Saliency Indicator: {salience_indicator}")

        if self.derive_temporal_grounding_summary:
            temp_grounding_summary = api.summary_from_noisy_perceptual_data(
                action_captions=root_node.state.spatial_node_state.action_captions,
                question="",
                words=self.num_words_temporal_grounding_summary,
            )
            root_node.state.spatial_node_state.temporal_grounding_summary = temp_grounding_summary
            print("Temporal Grounding Summary derived.")
            logging.debug(f"Temporal Grounding Summary: {temp_grounding_summary}")

        root_node.state.derived = True

    def get_num_words_for_summary(self, len_video_sec: float, total_num_words_summary: int, min_num_words_summary: int) -> int:
        word_count = max(total_num_words_summary // self.normalization_video_length * len_video_sec, min_num_words_summary)
        while word_count % 10 != 0:
            word_count -= 1
        
        return word_count
# #######################################################
# #######################################################
# Completion given Half of the code
# \n \n Now i will give you the part of one class which i have finished, help me to complete it based on the above requirment and description and give me the completed code finally.
# #######################################################
# #######################################################
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
    ) -> None:
        """
        Initialize the DeriveRootNodeState class with derived properties related to action captions, object detections, and temporal grounding summaries.
        """
        super(DeriveRootNodeState, self).__init__()

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

    def _execute(self, graph: Optional[Graph], api: Optional[API], target=None) -> None:
        """Execute operations required to derive various states."""
        root_node = graph.root
        len_video_sec = len(root_node.state.video_clip) / float(root_node.state.video_clip.sampled_fps)

        if self.derive_action_captions:
            try:
                root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(video_clip=root_node.state.video_clip)
                logger.info(f"Derived action captions for root node {root_node}")
                logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")
            except Exception as e:
                print(e)

        if self.derive_action_captions_summary:
            try:
                root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(action_captions=root_node.state.spatial_node_state.action_captions, object_detections=[], temporal_grounding=[])
                logger.info(f"Generated action captions summary for root node {root_node}")
                logger.debug(f"Action Captions Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")
            except Exception as e:
                print(e)

        if self.derive_object_detections:
            pass
            
        if self.derive_object_detections_summary:
            pass

        if self.derive_temporal_grounding:
            try:
                foreground_indicators, relevance_indicators, salience_indicators = api.get_temporal_grounding_from_video_clip_and_text(video_clip=root_node.state.video_clip, question="What actions were performed?")
                root_node.state.spatial_node_state.temporal_node_state.foreground_indicators = foreground_indicators
                root_node.state.spatial_node_state.temporal_node_state.relevance_indicators = relevance_indicators
                root_node.state.spatial_node_state.temporal_node_state.salience_indicators = salience_indicators
                logger.info(f"Temporally grounded root node {root_node} successfully.")
                logger.debug(f"Foreground Indicators\n{foreground_indicators}\nRelevance Indicators\n{relevance_indicators}\nSalience Indicators\n{salience_indicators}")
            except Exception as e:
                print(e)

        if self.derive_temporal_grounding_summary:
            try:
                root_node.state.spatial_node_state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[], object_detections=[], temporal_grounding=[root_node.state.spatial_node_state.temporal_node_state.get_textual_temporal_grounding()], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip.question, words="")
                logger.info(f"Generated temporal grounding summary for root node {root_node}")
                logger.debug(f"Temporal Grounding Summary:\n{root_node.state.spatial_node_state.temporal_node_state.temporal_grounding_summary}")
            except Exception as e:
                print(e)

        root_node.state.derived = True

    def get_num_words_for_summary(self, len_video_sec, total_num_words_summary, min_num_words_summary):  
        if self.normalization_video_length != -1:
            whole_video_word_contigent = round((float(total_num_words_summary)/self.normalization_video_length)*len_video_sec)
            while whole_video_word_contigent % 10 > 9:
                whole_video_word_contigent -= 1
            while whole_video_word_contigent < min_num_words_summary:
                whole_video_word_contigent += 1
            return max(min_num_words_summary, whole_video_word_contigent)
        else:
            return total_num_words_summary
# #######################################################
# #######################################################
# Filling blanks in the code
# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description. Give me the completed code finally.
# #######################################################
# #######################################################
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

    def _execute(self, graph: Optional[Graph], api: Optional[API], target=None) -> None:
        root_node = graph.root
        len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

        if self.derive_action_captions:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(video_clip=root_node.state.video_clip)

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(action_captions=root_node.state.spatial_node_state.action_captions, object_detections=[], temporal_grounding=[], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_action_captions_summary))

            logger.info(f"Derived action caption summary for root node {root_node}.")
            logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

        if self.derive_object_detections:
            # use object detection API function to get object detections from the video clip
            root_node.state.spatial_node_state.object_detections = api.get_unspecific_objects_from_video_clip(video_clip=root_node.state.video_clip)

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            # use object detection summary API function to summarize all object detections to get a universal video summary
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[], object_detections=root_node.state.spatial_node_state.get_textual_object_list(), temporal_grounding=[], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_object_detections_summary, min_num_words_summary=self.min_num_words_object_detections_summary))

            logger.info(f"Derived object detection summary for root node {root_node}.")
            logger.debug(f"Object Detection Summary:\n{root_node.state.spatial_node_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            # use temporal grounding API function to get temporal grounding from the video clip and the question
            temporal_grounding = api.get_temporal_grounding_from_video_clip_and_text(video_clip=root_node.state.video_clip, text=root_node.state.task.question)

            root_node.state.temporal_node_state.foreground_indicators = ["foreground_indicators"]
            root_node.state.temporal_node_state.relevance_indicators = temporal_grounding['relevance_indicators']
            root_node.state.temporal_node_state.salience_indicators = temporal_grounding['salience']

            logger.info(f"Derived temporal grounding for root node {root_node}.")
            logger.debug(f"Temporal Grounding:\n{root_node.state.temporal_node_state.get_textual_temporal_grounding()}")

        if self.derive_temporal_grounding_summary:
            # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
            root_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[], object_detections=[], temporal_grounding=[root_node.state.temporal_node_state.temporal_grounding_summary()], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, options=root_node.state.task.options, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_temporal_grounding_summary, min_num_words_summary=self.min_num_words_temporal_grounding_summary))

            logger.info(f"Derived temporal grounding summary for root node {root_node}.")
            logger.debug(f"Temporal Grounding Summary:\n{root_node.state.temporal_node_state.temporal_grounding_summary}")

        root_node.state.derived = True

        logger.info(f"Derived universal node state for root node {root_node}.")
        logger.debug(f"The following universal node state has been derived:{root_node.state}")
        logger.info(f"Executed root state derivation operation: DeriveUniversalState")

    def get_num_words_for_summary(
            self,
            len_video_sec,
            total_num_words_summary,
            min_num_words_summary
    ):
        # do not normalize the number of words for the summary if the normalization video length is -1
        if self.normalization_video_length == -1:
            num_words = total_num_words_summary
        else:
            # calculate the number of words for the summary (proportional to the video length in seconds)
            # calculate the word contingent for the whole video
            whole_video_word_contingent = ((total_num_words_summary / self.normalization_video_length)) * len_video_sec
            # make multiple of 10 for better LLM readability
            num_words_mul_ten = round((whole_video_word_contingent / 10))
            num_words = num_words_mul_ten * 10
            # clip the number of words to the minimum number of words
            num_words = max(num_words, min_num_words_summary)
        return num_words