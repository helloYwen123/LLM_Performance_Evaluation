
 # GPT-o1
 # Ground-Truth
 
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
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices[0]
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
                logger.debug(f"Action Captions:\n{derivable_node.state.spatial_node_state.action_captions}")

            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action caption summary: {num_words}")

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

                logger.info(f"Derived action caption summary for node {derivable_node}.")
                logger.debug(
                    f"Action Caption Summary:\n{derivable_node.state.spatial_node_state.action_captions_summary}")

            if self.derive_object_detections:
                # inherit the object detections from the root node
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                                              start_list_index:end_list_index + 1]

                # set the object detections of the derivable node
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections

                logger.info(f"Inherited object detections from root node for node {derivable_node}.")
                logger.debug(f"Object Detections:\n"
                             f"{derivable_node.state.spatial_node_state.object_detections}")

            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
                logger.debug(f"Number of words for object detections summary: {num_words}")

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

                logger.info(f"Derived object detection summary for node {derivable_node}.")
                logger.debug(f"Object Detection Summary:\n"
                             f"{derivable_node.state.spatial_node_state.object_detections_summary}")

            if self.derive_temporal_grounding:
                # inherit the temporal grounding from the root node
                derivable_node.state.temporal_node_state.foreground_indicators = root_state.temporal_node_state.foreground_indicators[
                                                                                 start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.relevance_indicators = root_state.temporal_node_state.relevance_indicators[
                                                                                start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.salience_indicators = root_state.temporal_node_state.salience_indicators[
                                                                               start_list_index:end_list_index + 1]

                logger.info(f"Inherited temporal grounding from root node for node {derivable_node}.")
                logger.debug(f"Temporal Grounding:\n"
                             f"{derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
                logger.debug(f"Number of words for temporal grounding summary: {num_words}")

                # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
                self.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,  # will be used if it is given in the prompt template
                    options=derivable_node.state.task.options,  # will be used if they are given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logger.info(f"Derived temporal grounding summary for node {derivable_node}.")
                logger.debug(f"Temporal Grounding Summary:\n"
                             f"{derivable_node.state.temporal_node_state.temporal_grounding_summary}")

            derivable_node.state.derived = True

            logger.info(f"Derived node state for node {derivable_node}.")
            logger.debug(f"The following state has been derived:\n{derivable_node.state}")

        logger.info(f"Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(
            self,
            len_clip_sec,
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
        # calculate the exact number of words for the clip
        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        # make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(num_words_exact / 10))) * 10
        # clip the number of words to the minimum number of words
        num_words = max(num_words_mul_ten, min_num_words_summary)
        return num_words
    
#################################################################################################

# Prompt： # here, you are an very experienced Programmer, who is very good at programming under others' instruction and can also complete code with many blank space or gaps very well you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# Now please help me to write a class named DeriveNodeStates, it derives from the other class Operation, its initialization function has following inputs:
# derive_action_captions: bool, derive_action_captions_summary: bool,total_num_words_action_captions_summary: int,min_num_words_action_captions_summary: int,derive_object_detections: bool,derive_object_detections_summary: bool,total_num_words_object_detections_summary: int,min_num_words_object_detections_summary: int,derive_temporal_grounding: bool,derive_temporal_grounding_summary: bool,total_num_words_temporal_grounding_summary: int,min_num_words_temporal_grounding_summary: int,normalization_video_length: int = 180
# the initialization parts are: self.derive_action_captions = derive_action_captions or derive_action_captions_summary,self.derive_action_captions_summary = derive_action_captions_summary,self.total_num_words_action_captions_summary = total_num_words_action_captions_summary,self.min_num_words_action_captions_summary = min_num_words_action_captions_summary
# self.derive_object_detections = derive_object_detections or derive_object_detections_summary,self.derive_object_detections_summary = derive_object_detections_summary,self.total_num_words_object_detections_summary = total_num_words_object_detections_summary,self.min_num_words_object_detections_summary = min_num_words_object_detections_summary
# self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary,self.derive_temporal_grounding_summary = derive_temporal_grounding_summary,self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary,self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary,self.normalization_video_length = normalization_video_length
# after this initallization is the first member function named _execute, it has inputs:graph: Optional[Graph], api: Optional[API], target=Optional[Node], and function return none.
# get the value from graph.root.state and create the variable root_state. and if the root_state.derived is not true, then reports error msg "Root node state has not been derived yet."
# create variable derivable_nodes from graph.get_derivable_nodes(). Iteratively get element in deriabled_nodes. Firstly, find and set the variable for finding start and end indices of the video clip， the form like total_start_frame_index = derivable_node.state.video_clip.sampled_indices[0], same for the last index
# then get the list index i####n the sampled indices of the root node state video via start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index) , same in end_list_index.
# then get the length of the clip and the whole video in seconds respectively, the length can be calculated with the num of frame devide by sample_fps, these can be like len(derivable_node.state.video_clip) and derivable_node.state.video_clip.sampled_fps. the whole video is from root_state.
# above all process will be repectively followed by logger debug
# if self.derive_action_captions is true then inherit the action captions from the root node root_state.spatial_node_state.action_captions[start_list_index:end_list_index + 1]
# then set the action captions of the derivable node: derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions. Following is logger info and debug for action caption and derivable_node.
# if self.derive_action_captions_summary is true, then call the member function self.get_num_words_for_summary to get num of words then logger.debug for it,then use action caption summary API function (api.get_summary_from_noisy_perceptive_data,which has inputs: action_captions=derivable_node.state.spatial_node_state.action_captions,object_detections=[],temporal_grounding=[],interleaved_action_captions_and_object_detections=[],video_clip=derivable_node.state.video_clip,question=derivable_node.state.task.question,words=num_words ,to summarize all action captions to get a clip summary and is followed by logger.info and logger.debug
# if self.derive_object_detections is true, then inherit the clip_node_object_detections with the same way as one for action_caption. Firstly get the variable clip_node_object_detections and assign it to derivable_node.state.spatial_node_state.object_detections, followed by logger.info and logger.debug.
# if self.derive_object_detections_summary is true , then also call the self.get_num_words_for_summary to get the num of words, followed by step of logger.debug
# next use object detection summary API function(api.get_summary_from_noisy_perceptive_data, which has different inputs from actionscaption.:action_captions=[],object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),temporal_grounding=[],interleaved_action_captions_and_object_detections=[],video_clip=derivable_node.state.video_clip,question=derivable_node.state.task.question,words=num_words) to summarize all object detections to get a universal video summary.
# then is followed with logger.info and logger.debug for derivable_node and object detection summary
# if self.derive_temporal_grounding is true，then inherit the temporal grounding from the root node including derivable_node.state.temporal_node_state.foreground_indicators, derivable_node.state.temporal_node_state.relevance_indicators,derivable_node.state.temporal_node_state.salience_indicators,in a way similar to actioncaption and objectdetection root_state.temporal_node_state. for call respectively foreground_indicators,relevance_indicators,salience_indicators
# above is followed by logger.info for derivable_node and logger.debug for derivable_node.state.temporal_node_state.get_textual_temporal_grounding()
# if self.derive_temporal_grounding_summary is true then we call self.get_num_words_for_summary to return num_words and through logger.debug for num_words
# use temporal grounding summary API function(api.get_summary_from_noisy_perceptive_data, which has inputs like action_captions=[],object_detections=[],temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],interleaved_action_captions_and_object_detections=[],video_clip=derivable_node.state.video_clip,question=derivable_node.state.task.question,options=derivable_node.state.task.options, words=num_words) to summarize all temporal grounding to get a universal video summary
# then followed by logger.info and logger.debug
# set derivable_node.state.derived = True, then logger.info derivable_node and erivable_node.state. At last step of function _execute is logger.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")
# the seconde function is named get_num_words_for_summary whose inputs are len_clip_sec,len_video_sec,total_num_words_summary,min_num_words_summary. if self.normalization_video_length = -1, then set total_num_words_summary to whole_video_word_contingent
# ohterwise  whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
# then calculate the exact number of words for the clip which is num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)));
# make multiple of 10 for better LLM readability transfering num_words_exact to num_words_mul_ten. Because inputs has min_num_words_summary, which ensures the num_words is maximun of num_words_mul_ten and min_num_words_summary

#####################################################################################################
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
        """
        Initializes the DeriveNodeStates operation with various flags and parameters.

        Args:
            derive_action_captions (bool): Flag to derive action captions.
            derive_action_captions_summary (bool): Flag to derive summary for action captions.
            total_num_words_action_captions_summary (int): Total number of words for action captions summary.
            min_num_words_action_captions_summary (int): Minimum number of words for action captions summary.
            derive_object_detections (bool): Flag to derive object detections.
            derive_object_detections_summary (bool): Flag to derive summary for object detections.
            total_num_words_object_detections_summary (int): Total number of words for object detections summary.
            min_num_words_object_detections_summary (int): Minimum number of words for object detections summary.
            derive_temporal_grounding (bool): Flag to derive temporal grounding.
            derive_temporal_grounding_summary (bool): Flag to derive summary for temporal grounding.
            total_num_words_temporal_grounding_summary (int): Total number of words for temporal grounding summary.
            min_num_words_temporal_grounding_summary (int): Minimum number of words for temporal grounding summary.
            normalization_video_length (int, optional): Normalization length for video in seconds. Defaults to 180.
        """
        super().__init__()

        # Initialize action captions related attributes
        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        # Initialize object detections related attributes
        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        # Initialize temporal grounding related attributes
        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        # Initialize normalization attribute
        self.normalization_video_length = normalization_video_length

        logger.debug("DeriveNodeStates initialized with the following parameters:")
        logger.debug(f"derive_action_captions: {self.derive_action_captions}")
        logger.debug(f"derive_action_captions_summary: {self.derive_action_captions_summary}")
        logger.debug(f"total_num_words_action_captions_summary: {self.total_num_words_action_captions_summary}")
        logger.debug(f"min_num_words_action_captions_summary: {self.min_num_words_action_captions_summary}")
        logger.debug(f"derive_object_detections: {self.derive_object_detections}")
        logger.debug(f"derive_object_detections_summary: {self.derive_object_detections_summary}")
        logger.debug(f"total_num_words_object_detections_summary: {self.total_num_words_object_detections_summary}")
        logger.debug(f"min_num_words_object_detections_summary: {self.min_num_words_object_detections_summary}")
        logger.debug(f"derive_temporal_grounding: {self.derive_temporal_grounding}")
        logger.debug(f"derive_temporal_grounding_summary: {self.derive_temporal_grounding_summary}")
        logger.debug(f"total_num_words_temporal_grounding_summary: {self.total_num_words_temporal_grounding_summary}")
        logger.debug(f"min_num_words_temporal_grounding_summary: {self.min_num_words_temporal_grounding_summary}")
        logger.debug(f"normalization_video_length: {self.normalization_video_length}")

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node'] = None
    ) -> None:
        """
        Executes the state derivation operation on derivable nodes within the graph.

        Args:
            graph (Optional[Graph]): The graph containing nodes.
            api (Optional[API]): The API used for summarization.
            target (Optional[Node], optional): The target node. Defaults to None.
        """
        if graph is None:
            logger.error("Graph is None.")
            return

        root_state = graph.root.state
        logger.debug(f"Retrieved root state: {root_state}")

        if not getattr(root_state, 'derived', False):
            logger.error("Root node state has not been derived yet.")
            return

        derivable_nodes = graph.get_derivable_nodes()
        logger.debug(f"Found {len(derivable_nodes)} derivable nodes.")

        for derivable_node in derivable_nodes:
            logger.debug(f"Processing derivable node: {derivable_node}")

            # Extract start and end frame indices
            video_clip = derivable_node.state.video_clip
            sampled_indices = video_clip.sampled_indices
            total_start_frame_index = sampled_indices[0]
            total_end_frame_index = sampled_indices[-1]
            logger.debug(f"Start frame index: {total_start_frame_index}, End frame index: {total_end_frame_index}")

            # Find list indices in root state
            try:
                start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
                end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
                logger.debug(f"Start list index: {start_list_index}, End list index: {end_list_index}")
            except ValueError as e:
                logger.error(f"Frame index not found in root state: {e}")
                continue

            # Calculate lengths
            len_clip_frames = len(sampled_indices)
            sample_fps = video_clip.sampled_fps
            len_clip_sec = len_clip_frames / sample_fps
            len_video_sec = len(root_state.video_clip.sampled_indices) / root_state.video_clip.sampled_fps
            logger.debug(f"Clip length: {len_clip_sec} seconds, Video length: {len_video_sec} seconds")

            # Action Captions
            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions
                logger.info(f"Action captions derived for node {derivable_node}.")
                logger.debug(f"Action captions: {clip_node_action_captions}")

            # Action Captions Summary
            if self.derive_action_captions_summary and api:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec, len_video_sec,
                    self.total_num_words_action_captions_summary,
                    self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action captions summary: {num_words}")

                action_caption_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.action_captions_summary = action_caption_summary
                logger.info(f"Action captions summary derived for node {derivable_node}.")
                logger.debug(f"Action captions summary: {action_caption_summary}")

            # Object Detections
            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections
                logger.info(f"Object detections derived for node {derivable_node}.")
                logger.debug(f"Object detections: {clip_node_object_detections}")

            # Object Detections Summary
            if self.derive_object_detections_summary and api:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec, len_video_sec,
                    self.total_num_words_object_detections_summary,
                    self.min_num_words_object_detections_summary
                )
                logger.debug(f"Number of words for object detections summary: {num_words}")

                object_detection_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.object_detections_summary = object_detection_summary
                logger.info(f"Object detections summary derived for node {derivable_node}.")
                logger.debug(f"Object detections summary: {object_detection_summary}")

            # Temporal Grounding
            if self.derive_temporal_grounding:
                temporal_state = root_state.temporal_node_state
                derivable_node.state.temporal_node_state.foreground_indicators = temporal_state.foreground_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.relevance_indicators = temporal_state.relevance_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.salience_indicators = temporal_state.salience_indicators[start_list_index:end_list_index + 1]
                logger.info(f"Temporal grounding derived for node {derivable_node}.")
                logger.debug(f"Temporal grounding: {derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            # Temporal Grounding Summary
            if self.derive_temporal_grounding_summary and api:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec, len_video_sec,
                    self.total_num_words_temporal_grounding_summary,
                    self.min_num_words_temporal_grounding_summary
                )
                logger.debug(f"Number of words for temporal grounding summary: {num_words}")

                temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=derivable_node.state.temporal_node_state.get_textual_temporal_grounding(),
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )
                derivable_node.state.temporal_node_state.temporal_grounding_summary = temporal_grounding_summary
                logger.info(f"Temporal grounding summary derived for node {derivable_node}.")
                logger.debug(f"Temporal grounding summary: {temporal_grounding_summary}")

            # Mark node as derived
            derivable_node.state.derived = True
            logger.info(f"Node {derivable_node} marked as derived.")
            logger.debug(f"Derived state: {derivable_node.state}")

        logger.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(
        self,
        len_clip_sec: float,
        len_video_sec: float,
        total_num_words_summary: int,
        min_num_words_summary: int
    ) -> int:
        """
        Calculates the number of words for the summary based on video lengths and normalization.

        Args:
            len_clip_sec (float): Length of the clip in seconds.
            len_video_sec (float): Total length of the video in seconds.
            total_num_words_summary (int): Total number of words allocated for the summary.
            min_num_words_summary (int): Minimum number of words for the summary.

        Returns:
            int: The calculated number of words for the summary.
        """
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
            logger.debug("Normalization video length is -1. Using total_num_words_summary as is.")
        else:
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
            logger.debug(f"Calculated whole_video_word_contingent: {whole_video_word_contingent}")

        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        logger.debug(f"Exact number of words for clip: {num_words_exact}")

        # Make it a multiple of 10 for better readability
        num_words_mul_ten = max((num_words_exact // 10) * 10, min_num_words_summary)
        if num_words_exact % 10 >= 5:
            num_words_mul_ten += 10
        num_words_mul_ten = max(num_words_mul_ten, min_num_words_summary)
        logger.debug(f"Number of words after rounding to multiple of ten: {num_words_mul_ten}")

        return num_words_mul_ten
###############################################################################################################################################
# Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the following requirment and description.
# Please do not start when i still don't show you my code and requrement
# 
###################################################################################################################################################
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

        # Action Captions Initialization
        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        # Object Detections Initialization
        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        # Temporal Grounding Initialization
        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        # Normalization Video Length
        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional['Graph'], api: Optional['API'], target: Optional['Node']) -> None:
        if graph is None or api is None:
            err_msg = "Graph and API must be provided."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Get the root node state from the graph
        root_state = graph.root.state
        logger.debug("Retrieved root node state.")

        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        derivable_nodes = graph.get_derivable_nodes()
        logger.debug(f"Found {len(derivable_nodes)} derivable nodes.")

        for derivable_node in derivable_nodes:
            logger.debug(f"Processing derivable node: {derivable_node}")

            # Find start and end indices of the video clip
            sampled_indices = derivable_node.state.video_clip.sampled_indices
            if not sampled_indices:
                logger.error(f"Derivable node {derivable_node} has no sampled indices.")
                continue

            total_start_frame_index = sampled_indices[0]
            total_end_frame_index = sampled_indices[-1]
            logger.debug(f"Total start frame index: {total_start_frame_index}")
            logger.debug(f"Total end frame index: {total_end_frame_index}")

            # Get the list index in the sampled indices of the root node state video
            try:
                start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
                end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
                logger.debug(f"Start list index: {start_list_index}")
                logger.debug(f"End list index: {end_list_index}")
            except ValueError as e:
                logger.error(f"Frame index not found in root sampled indices: {e}")
                continue

            # Get the length of the clip and the whole video in seconds
            len_clip_sec = len(derivable_node.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logger.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logger.debug(f"Length of the video in seconds: {len_video_sec}")

            # Derive Action Captions
            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions
                logger.info(f"Inherited action captions from root node for node {derivable_node}.")
                logger.debug(f"Action captions: {clip_node_action_captions}")

            # Derive Action Captions Summary
            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec,
                    len_video_sec,
                    self.total_num_words_action_captions_summary,
                    self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action captions summary: {num_words}")

                action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.action_captions_summary = action_captions_summary
                logger.info(f"Derived action captions summary for node {derivable_node}.")
                logger.debug(f"Action captions summary: {action_captions_summary}")

            # Derive Object Detections
            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections
                logger.info(f"Inherited object detections from root node for node {derivable_node}.")
                logger.debug(f"Object detections: {clip_node_object_detections}")

            # Derive Object Detections Summary
            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec,
                    len_video_sec,
                    self.total_num_words_object_detections_summary,
                    self.min_num_words_object_detections_summary
                )
                logger.debug(f"Number of words for object detections summary: {num_words}")

                object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.object_detections_summary = object_detections_summary
                logger.info(f"Derived object detections summary for node {derivable_node}.")
                logger.debug(f"Object detections summary: {object_detections_summary}")

            # Derive Temporal Grounding
            if self.derive_temporal_grounding:
                temporal_state = root_state.temporal_node_state
                derivable_node.state.temporal_node_state.foreground_indicators = temporal_state.foreground_indicators[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.temporal_node_state.relevance_indicators = temporal_state.relevance_indicators[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.temporal_node_state.salience_indicators = temporal_state.salience_indicators[
                    start_list_index:end_list_index + 1
                ]
                logger.info(f"Inherited temporal grounding from root node for node {derivable_node}.")
                logger.debug(f"Temporal grounding: {derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            # Derive Temporal Grounding Summary
            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec,
                    len_video_sec,
                    self.total_num_words_temporal_grounding_summary,
                    self.min_num_words_temporal_grounding_summary
                )
                logger.debug(f"Number of words for temporal grounding summary: {num_words}")

                temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=derivable_node.state.temporal_node_state.get_textual_temporal_grounding(),
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )
                derivable_node.state.temporal_node_state.temporal_grounding_summary = temporal_grounding_summary
                logger.info(f"Derived temporal grounding summary for node {derivable_node}.")
                logger.debug(f"Temporal grounding summary: {temporal_grounding_summary}")

            # Mark the node as derived
            derivable_node.state.derived = True
            logger.info(f"Marked node {derivable_node} as derived.")
            logger.debug(f"Derived state: {derivable_node.state}")

        # Final log after processing all nodes
        logger.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(
            self,
            len_clip_sec: float,
            len_video_sec: float,
            total_num_words_summary: int,
            min_num_words_summary: int
    ) -> int:
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
            logger.debug("Normalization video length is -1; using total_num_words_summary directly.")
        else:
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
            logger.debug(f"Calculated whole_video_word_contingent: {whole_video_word_contingent}")

        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        logger.debug(f"Exact number of words for the clip: {num_words_exact}")

        # Make it a multiple of 10 for better readability
        num_words_mul_ten = max(
            (num_words_exact // 10) * 10,
            min_num_words_summary
        )
        if num_words_exact % 10 >= 5:
            num_words_mul_ten += 10  # Round up if remainder >= 5

        num_words_final = max(num_words_mul_ten, min_num_words_summary)
        logger.debug(f"Final number of words for summary (multiple of ten): {num_words_final}")

        return num_words_final
    ####################################################################################################################################################################    
    
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

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node] = None
    ) -> None:
        if graph is None or api is None:
            logger.error("Graph and API must not be None.")
            raise ValueError("Graph and API must not be None.")

        # Get the root node state from the graph
        root_state = graph.root.state

        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Get derivable nodes
        derivable_nodes = graph.get_derivable_nodes()
        logger.debug(f"Found {len(derivable_nodes)} derivable nodes.")

        for derivable_node in derivable_nodes:
            # Find start and end indices of the video clip
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices[0]
            total_end_frame_index = derivable_node.state.video_clip.sampled_indices[-1]
            logger.debug(f"Total start frame index: {total_start_frame_index}")
            logger.debug(f"Total end frame index: {total_end_frame_index}")

            # Get the list indices in the sampled indices of the root node state video
            try:
                start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
                end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
                logger.debug(f"Start list index: {start_list_index}")
                logger.debug(f"End list index: {end_list_index}")
            except ValueError as e:
                logger.error(f"Frame index not found in root sampled indices: {e}")
                raise

            # Get the length of the clip and the whole video in seconds
            len_clip_sec = len(derivable_node.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logger.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logger.debug(f"Length of the video in seconds: {len_video_sec}")

            # Derive Action Captions
            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions

                logger.info(f"Inherited action captions from root node for node {derivable_node}.")
                logger.debug(f"Action Captions:\n{derivable_node.state.spatial_node_state.action_captions}")

            # Derive Action Captions Summary
            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action caption summary: {num_words}")

                action_caption_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.action_captions_summary = action_caption_summary

                logger.info(f"Derived action caption summary for node {derivable_node}.")
                logger.debug(f"Action Caption Summary:\n{derivable_node.state.spatial_node_state.action_captions_summary}")

            # Derive Object Detections
            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections

                logger.info(f"Inherited object detections from root node for node {derivable_node}.")
                logger.debug(f"Object Detections:\n{derivable_node.state.spatial_node_state.object_detections}")

            # Derive Object Detections Summary
            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
                logger.debug(f"Number of words for object detections summary: {num_words}")

                object_detection_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )
                derivable_node.state.spatial_node_state.object_detections_summary = object_detection_summary

                logger.info(f"Derived object detection summary for node {derivable_node}.")
                logger.debug(f"Object Detection Summary:\n{derivable_node.state.spatial_node_state.object_detections_summary}")

            # Derive Temporal Grounding
            if self.derive_temporal_grounding:
                derivable_node.state.temporal_node_state.foreground_indicators = root_state.temporal_node_state.foreground_indicators[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.temporal_node_state.relevance_indicators = root_state.temporal_node_state.relevance_indicators[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.temporal_node_state.salience_indicators = root_state.temporal_node_state.salience_indicators[
                    start_list_index:end_list_index + 1
                ]

                logger.info(f"Inherited temporal grounding from root node for node {derivable_node}.")
                logger.debug(f"Temporal Grounding:\n{derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

            # Derive Temporal Grounding Summary
            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
                logger.debug(f"Number of words for temporal grounding summary: {num_words}")

                temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=derivable_node.state.temporal_node_state.get_textual_temporal_grounding(),
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )
                derivable_node.state.temporal_node_state.temporal_grounding_summary = temporal_grounding_summary

                logger.info(f"Derived temporal grounding summary for node {derivable_node}.")
                logger.debug(f"Temporal Grounding Summary:\n{derivable_node.state.temporal_node_state.temporal_grounding_summary}")

            # Mark the node as derived
            derivable_node.state.derived = True

            logger.info(f"Derived node state for node {derivable_node}.")
            logger.debug(f"The following state has been derived:\n{derivable_node.state}")

        logger.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(
        self,
        len_clip_sec: float,
        len_video_sec: float,
        total_num_words_summary: int,
        min_num_words_summary: int
    ) -> int:
        """
        Calculate the number of words for the summary based on the clip and video lengths.
        """
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
            logger.debug("Normalization video length is -1. Using total_num_words_summary as is.")
        else:
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
            logger.debug(f"Calculated whole_video_word_contingent: {whole_video_word_contingent}")

        # Calculate the exact number of words for the clip
        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        logger.debug(f"Exact number of words for the clip: {num_words_exact}")

        # Make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(num_words_exact / 10))) * 10
        logger.debug(f"Number of words rounded to multiple of ten: {num_words_mul_ten}")

        # Ensure the number of words is at least the minimum required
        num_words = max(num_words_mul_ten, min_num_words_summary)
        logger.debug(f"Final number of words for summary: {num_words}")

        return num_words
    
    
  