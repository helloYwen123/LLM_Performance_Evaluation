# #######################################################
# #######################################################
# Based on Description to generate code
# #######################################################
# #######################################################

class DeriveRootNodeState(Operation):
    def __init__(self, derive_action_captions=False, derive_action_captions_summary=False, num_words_action_captions_summary=None, min_num_words_action_captions_summary=None, derive_object_detections=False, derive_object_detections_summary=False, num_words_object_detections_summary=None, min_num_words_object_detections_summary=None, derive_temporal_grounding=False, derive_temporal_grounding_summary=False, num_words_temporal_grounding_summary=None, min_num_words_temporal_grounding_summary=None, normalization_video_length=-1) -> None:
        super().__init__()
        
        # Initialize derived state flags based on constructor arguments
        self.derive_action_captions = derive_action_captions
        self.derive_action_captions_summary = derive_action_captions_summary
        self.num_words_action_captions_summary = (num_words_action_captions_summary or 5 * max(min_num_words_action_captions_summary // 10 + 1, 1)) if min_num_words_action_captions_summary else None
        
        self.derive_object_detections = derive_object_detections
        self.derive_object_detections_summary = derive_object_detections_summary
        self.num_words_object_detections_summary = (num_words_object_detections_summary or 5 * max(min_num_words_object_detections_summary // 10 + 1, 1)) if min_num_words_object_detections_summary else None

        self.derive_temporal_grounding = derive_temporal_grounding
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.num_words_temporal_grounding_summary = (num_words_temporal_grounding_summary or 5 * max(min_num_words_temporal_grounding_summary // 10 + 1, 1)) if min_num_words_temporal_grounding_summary else None

        self.normalization_video_length = normalization_video_length
    
    def _execute(self, graph: Optional['Graph'], api: Optional['API'], target: Optional['Node']) -> None:
        assert isinstance(graph, Graph)

        try:
            root_node = graph.root_node
            sample_rate = float(root_node.state.video_clip.sampled_fps)
            
            # Calculate duration of each video segment
            len_video_sec = round(len(root_node.state.video_clip)/sample_rate)
            
            print(f"Duration of {len_video_sec} sec")
            
            if self.derive_action_captions:
                action_captions = api.get_action_captions(root_node.state.video_clip)
                root_node.state.spatial_node_state.action_captions = action_captions
            
            if self.derive_action_captions_summary:
                action_captions_summary = api.summary_from_noisy_perceptual_data(action_captions=root_node.state.spatial_node_state.action_captions)
                root_node.state.spatial_node_state.action_captions_summary = action_captions_summary
                
            if self.derive_object_detections:
                object_detections = api.get_unspecified_objects_from_video_clip(
                    action_captions=root_node.state.spatial_node_state.action_captions,
                    object_detections=True,
                    temp_grounding=self.derive_temporal_grounding,
                    interleaved_action_captions_and_object_detections=True,
                    video_clip=root_node.state.video_clip,
                    question="",
                    words=len_video_sec
                )
                
                root_node.state.spatial_node_state.object_detections = object_detections
            
            if self.derive_temporal_grounding:
                temporal_grounding = api.get_temporal_grounding_from_video_clip_and_text(video_clip=root_node.state.video_clip, text="")
                foreground_indicators, relevance_indicators, salience_indicators = temporal_grounding
                
                root_node.state.temporal_grounding_foreground_indicators = foreground_indicators
                root_node.state.temporal_grounding_relevance_indicators = relevance_indicators
                root_node.state.temporal_grounding_salience_indicators = salience_indicators
            
            if self.derive_temporal_grounding_summary:
                temporal_grounding_summary = api.summary_from_noisy_perceptual_data(temporal_grounding=temporal_grounding)
                root_node.state.spatial_node_state.temporal_grounding_summary = temporal_grounding_summary
            
            root_node.state.derived = True
            print("Derived successfully.")
        except Exception as e:
            raise ValueError(f"Error during derivation process: {e}")

    
    def get_num_words_for_summary(self, len_video_sec: int, total_num_words_summary: int | None, min_num_words_summary: int) -> int:
        if self.normalization_video_length == -1:
            word_count = total_num_words_summary
        else:
            normalized_total_num_words_summary = total_num_words_summary / self.normalization_video_length
            word_count = math.ceil(normalized_total_num_words_summary * len_video_sec)
        
        while word_count < min_num_words_summary:
            word_count += 10
        
        return word_count

# #######################################################
# #######################################################
# Completion given Half of the code
# \n Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the above requirment and description.
# \n\n class DeriveRootNodeState(Operation):
#     def __init__(
#             self,
#             derive_action_captions: bool,
#             derive_action_captions_summary: bool,
#             num_words_action_captions_summary: int,
#             min_num_words_action_captions_summary: int,
#             derive_object_detections: bool,
#             derive_object_detections_summary: bool,
#             num_words_object_detections_summary: int,
            
#         self.derive_action_captions = derive_action_captions or derive_action_captions_summary
#         self.derive_action_captions_summary = derive_action_captions_summary
#         self.num_words_action_captions_summary = num_words_action_captions_summary
#         self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

#         self.derive_object_detections = derive_object_detections or derive_object_detections_summary
#         self.derive_object_detections_summary = derive_object_detections_summary
#         self.num_words_object_detections_summary = num_words_object_detections_summary
#         self.min_num_words_object_detections_summary = min_num_words_object_detections_summary
        
        
        
#     def _execute(self, graph: Optional[Graph], api: Optional[API], target=Optional[Node]) -> None:
#         root_node = graph.root
#         len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

#         if self.derive_action_captions:
#             # use action caption summary API function to summarize all action captions to get a universal video summary
#             root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(
#                 video_clip=root_node.state.video_clip
#             )

#             logger.info(f"Derived action captions for root node {root_node}.")
#             logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

#         if self.derive_action_captions_summary:
#             # use action caption summary API function to summarize all action captions to get a universal video summary
#             root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
#                 action_captions=root_node.state.spatial_node_state.action_captions,
#                 object_detections=[],
#                 temporal_grounding=[],

        
#         if self.derive_temporal_grounding_summary:
#             # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
#             root_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
#                 action_captions=[],
#                 object_detections=[],
#                 temporal_grounding=[root_node.state.temporal_node_state.get_textual_temporal_grounding()],
#                 interleaved_action_captions_and_object_detections=[],
#                 video_clip=root_node.state.video_clip,
                
#     def get_num_words_for_summary(
#             self,
#             len_video_sec,
#             total_num_words_summary,
#             min_num_words_summary
#     ):
#         # do not normalize the number of words for the summary if the normalization video length is -1
#         if self.normalization_video_length == -1:
#             whole_video_word_contingent = total_num_words_summary
#         else:
#             # calculate the number of words for the summary (proportionate to the video length in seconds)
#             # calculate the word contingent for the whole video
#             whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
# #######################################################
# #######################################################
class DeriveRootNodeState(Operation):
    def __init__(self,
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
                 normalization_video_length: int = 180):

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
            # Use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(video_clip=root_node.state.video_clip)

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            # Use action caption summary API function to summarize all action captions to get a universal video summary
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(action_captions=root_node.state.spatial_node_state.action_captions,
                                                                                                                     object_detections=[],
                                                                                                                     temporal_grounding=[],

        if self.derive_object_detections:
            # Call API functions to extract objects detections summaries
            root_node.state.spatial_node_state.object_detections = api.get_unspecified_objects_from_video_clip(action_captions=root_node.state.spatial_node_state.action_captions,
                                                                                                               object_detections=[],
                                                                                                               temporal_grounding=[],
                                                                                                               interleaved_action_captions_and_object_detections=[],
                                                                                                               video_clip=root_node.state.video_clip,
                                                                                                               question="",
                                                                                                               words=self.num_words_object_detections_summary)

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            # Use object detection summary API function to summarize detected objects across different time intervals within each video segment
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[],
                                                                                                                      object_detections=root_node.state.spatial_node_state.object_detections,
                                                                                                                      temporal_grounding=[],
                                                                                                                      interleaved_action_captions_and_object_detections=[],
                                                                                                                      video_clip=root_node.state.video_clip,
                                                                                                                      question="")

        if self.derive_temporal_grounding:
            # Extract foreground indicators, relevance indicators, salience indicators through temporal grounding analysis
            fg_indicator, rel_indicator, saliency_indicator = api.get_temporal_grounding_from_video_clip_and_text(video_clip=root_node.state.video_clip,
                                                                                                                   text="")
            root_node.state.spatial_node_state.foreground_indicators.append(fg_indicator)
            root_node.state.spatial_node_state.relevance_indicators.append(rel_indicator)
            root_node.state.spatial_node_state.saliency_indicators.append(saliency_indicator)

            logger.info(f"Extracted temporal grounding features for root node {root_node}.")

        if self.derive_temporal_grounding_summary:
            # Summarize extracted temporal grounding features
            root_node.state.spatial_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[],
                                                                                                                       object_detections=[],
                                                                                                                       temporal_grounding=[],
                                                                                                                       interleaved_action_captions_and_object_detections=[],
                                                                                                                       video_clip=root_node.state.video_clip,
                                                                                                                       question="")


def get_num_words_for_summary(self, len_video_sec, total_num_words_summary, min_num_words_summary):
    """
    Calculate the number of words required per second for summarizing content while ensuring that at least `min_num_words_summary` can fit entirely inside the video duration (`len_video_sec`). If necessary, round up so there’s always enough space left over after accounting for minimum requirements.
    
    Args:
      @param len_video_sec: Length of the entire video file in seconds.
      @param total_num_words_summary: Total number of words available for generating the summary.
      @param min_num_words_summary: Minimum number of words expected to appear during generation process.
      
    Returns:
      Number_of_words_to_use: Calculated number of words to include in final summary.
    """

    if self.normalization_video_length != -1:
        # Normalize according to provided normalization factor
        normalized_total_num_words_summary = float(total_num_words_summary/self.normalization_video_length)
        # Ensure we don't go below our minimum requirement even post-normalization
        adjusted_normalized_total_num_words_summary = max(normalized_total_num_words_summary + ((normalized_total_num_words_summary-min_num_words_summary)/max(len_video_sec/5., .4)), min_num_words_summary)
        # Round down since we want integer values here
        num_words_to_use = math.floor(adjusted_normalized_total_num_words_summary*len_video_sec)
    else:
        # No normalization applied directly uses original value
        num_words_to_use = total_num_words_summary
    
    return num_words_to_use
    
# #######################################################
# #######################################################
# Filling blanks in the code
# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description.
# \n\n class DeriveRootNodeState(Operation):
#     def __init__(
#             self,
#             derive_action_captions: bool,
#             derive_action_captions_summary: bool,
#             num_words_action_captions_summary: int,
#             min_num_words_action_captions_summary: int,
#             derive_object_detections: bool,
#             derive_object_detections_summary: bool,
#             num_words_object_detections_summary: int,
#             min_num_words_object_detections_summary: int,
#             : bool,
#             derive_temporal_grounding_summary: ,
#             : int,
#             min_num_words_temporal_grounding_summary: ,
#             normalization_video_length: int = 
#     ):
#         super().__init__()

#         self. = derive_action_captions or derive_action_captions_summary
#         self.derive_action_captions_summary = derive_action_captions_summary
#         self.num_words_action_captions_summary = num_words_action_captions_summary
#         self.min_num_words_action_captions_summary = min_num_wordsaction_captions_summary

#         self.derive_object_detections = derive_object_detections or derive_object_detections_summary
#         self. = derive_object_detections_summary
#         self.num_words_object_detections_summary = num_words_object_detections_summary
#         self.min_num_words_object_detections_summary = 

#         self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
#         self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
#         self. = num_words_temporal_grounding_summary
#         self.min_num_words_temporal_grounding_summary = 

#         self.normalization_video_length = normalization_video_length

#     def _execute(self, graph: Optional[Graph], api: Optional[API], target=Optional[Node]) -> None:
#          = graph.root
#         len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

#         if self.derive_action_captions:
#             # use action caption summary API function to summarize all action captions to get a universal video summary
#             root_node.state..action_captions = api.get_action_captions_from_video_clip(
#                 video_clip=root_node.state.
#             )

#             logger.info(f"Derived action captions for root node {root_node}.")
#             logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

#         if self.:
#             # use action caption summary API function to summarize all action captions to get a universal video summary
#             root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
#                 action_captions=root_node.state.spatial_node_state.action_captions,
#                 object_detections=[],
#                 temporal_grounding=[],
#                 =[],
#                 video_clip=root_node.state.video_clip,
#                 question=root_node.state.task.question,  # will be used if it is given in the prompt template
#                 words=self.get_num_words_for_summary(
#                     len_video_sec=len_video_sec,
#                     total_num_words_summary=self.,
#                     =self.min_num_words_action_captions_summary
#                 )  # will be used if it is given in the prompt template
#             )

#             logger.info(f"Derived action caption summary for root node {}.")
#             logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

#         if self.derive_object_detections:
#             # use object detection API function to get object detections from the video clip
#             root_node.state.spatial_node_state. = api.get_unspecific_objects_from_video_clip(
#                 video_clip=root_node.state.)

#             logger.info(f"Derived object detections for root node {root_node}.")
#             logger.debug(f"Object Detections:\n"
#                          f"{root_node.state.spatial_node_state.object_detections}")

#         if self.:
#             # use object detection summary API function to summarize all object detections to get a universal video summary
#             root_node.state.spatial_node_state.object_detections_summary = api.(
#                 =[],
#                 object_detections=root_node.state.spatial_node_state.get_textual_object_list(),
#                 temporal_grounding=[],
#                 interleaved_action_captions_and_object_detections=[],
#                 video_clip=root_node.state.video_clip,
#                 question=root_node.state.task.question,  # will be used if it is given in the prompt template
#                 words=self.get_num_words_for_summary(
#                     len_video_sec=len_video_sec,
#                     total_num_words_summary=self.num_words_object_detections_summary,
#                     min_num_words_summary=self.min_num_words_object_detections_summary
#                 )  # will be used if it is given in the prompt template
#             )

#             logger.info(f"Derived object detection summary for root node {root_node}.")
#             logger.debug(f"Object Detection Summary:\n"
#                          f"{root_node.state.spatial_node_state.object_detections_summary}")

#         if self.derive_temporal_grounding:
#             # use temporal grounding API function to get temporal grounding from the video clip and the question
#              = api.get_temporal_grounding_from_video_clip_and_text(
#                 video_clip=root_node.state.video_clip,
#                 text=root_node.state.task.question
#             )

#             root_node.state.temporal_node_state.foreground_indicators = ["foreground_indicators"]
#             root_node.state.temporal_node_state.relevance_indicators = temporal_grounding["relevance_indicators"]
#             root_node.state.temporal_node_state.salience_indicators = temporal_grounding[""]

#             logger.info(f"Derived temporal grounding for root node {root_node}.")
#             logger.debug(f"Temporal Grounding:\n"
#                          f"{root_node.state.temporal_node_state.get_textual_temporal_grounding()}")

#         if self.:
#             # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
#             root_node.state.temporal_node_state. = api.get_summary_from_noisy_perceptive_data(
#                 action_captions=[],
#                 object_detections=[],
#                 temporal_grounding=[root_node.state.temporal_node_state.()],
#                 interleaved_action_captions_and_object_detections=[],
#                 =root_node.state.video_clip,
#                 question=root_node.state.task.,  # will be used if it is given in the prompt template
#                 =root_node.state.task.options,  # will be used if they are given in the prompt template
#                 words=self.(
#                     len_video_sec=len_video_sec,
#                     total_num_words_summary=self.,
#                     =self.min_num_words_temporal_grounding_summary
#                 )  # will be used if it is given in the prompt template
#             )

#             logger.info(f"Derived temporal grounding summary for root node {root_node}.")
#             logger.debug(f"Temporal Grounding Summary:\n"
#                          f"{root_node.state.temporal_node_state.temporal_grounding_summary}")

#          = True

#         logger.info(f"Derived universal node state for root node {root_node}.")
#         logger.debug(f"The following universal node state has been derived:\n{root_node.state}")
#         logger.info(f"Executed root state derivation operation: DeriveUniversalState")

#     def get_num_words_for_summary(
#             self,
#             len_video_sec,
#             ,
#             min_num_words_summary
#     ):
#         # do not normalize the number of words for the summary if the normalization video length is -1
#         if self.normalization_video_length == -1:
#              = total_num_words_summary
#         else:
#             # calculate the number of words for the summary (proportionate to the video length in seconds)
#             # calculate the word contingent for the whole video
#             whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * 
#         # make multiple of 10 for better LLM readability
#         num_words_mul_ten = (int(round( / 10))) * 10
#         # clip the number of words to the minimum number of words
#          = max(num_words_mul_ten, )
#         return 
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
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(video_clip=root_node.state.video_clip)

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(action_captions=root_node.state.spatial_node_state.action_captions, object_detections=[], temporal_grounding=[], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_action_captions_summary))

            logger.info(f"Derived action caption summary for root node {root_node}.")
            logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

        if self.derive_object_detections:
            root_node.state.spatial_node_state.object_detections = api.get_unspecific_objects_from_video_clip(video_clip=root_node.state.video_clip)

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[], object_detections=root_node.state.spatial_node_state.get_textual_object_list(), temporal_grounding=[], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_object_detections_summary, min_num_words_summary=self.min_num_words_object_detections_summary))

            logger.info(f"Derived object detection summary for root node {root_node}.")
            logger.debug(f"Object Detection Summary:\n{root_node.state.spatial_node_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            temp_grounding_result = api.get_temporal_grounding_from_video_clip_and_text(video_clip=root_node.state.video_clip, text=root_node.state.task.question)

            root_node.state.temporal_node_state.foreground_indicators = ['foreground_indicators']
            root_node.state.temporal_node_state.relevance_indicators = temp_grounding_result['relevance_indicators']
            root_node.state.temporal_node_state.salience_indicators = temp_grounding_result['salience_indicators']

            logger.info(f"Derived temporal grounding for root node {root_node}.")
            logger.debug(f"Temporal Grounding:\n{root_node.state.temporal_node_state.get_textual_temporal_grounding()}")

        if self.derive_temporal_grounding_summary:
            root_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(action_captions=[], object_detections=[], temporal_grounding=[temp_grounding_result['']], interleaved_action_captions_and_object_detections=[], video_clip=root_node.state.video_clip, question=root_node.state.task.question, options=root_node.state.task.options, words=self.get_num_words_for_summary(len_video_sec=len_video_sec, total_num_words_summary=self.num_words_temporal_grounding_summary, min_num_words_summary=self.min_num_words_temporal_grounding_summary))

            logger.info(f"Derived temporal grounding summary for root_node {root_node}.")
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
        if self.normalization_video_length == -1:
            num_words = total_num_words_summary
        else:
            whole_video_word_contingent = ((total_num_words_summary / self.normalization_video_length)) * len_video_sec
            num_words_mul_ten = round((whole_video_word_contingent + 5)/10)*10
            num_words = max(int(num_words_mul_ten // 10*10), min_num_words_summary)
        
        return num_words