# GPT-o1-Preview

## Ground Truth Code
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





# QUESTIONS ：

# here, you are an very experienced Programmer, 
# who is very good at programming under others' instruction 
# and can also complete code with many blank space or gaps very well
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.
# especially using Python completely satisfying the requirement.
# Next please you complete the code, 
# including classes and functions for my algorithm framework under my instruction and decription

# ANSWERS:

# Certainly! Please provide your algorithm framework and any specific instructions,
# and I'll complete the Python code with the necessary classes and functions to fully meet your requirements.


# QUESTIONS (only instruction)：

# help me to write a class framework called DeriveRootNodeState firstly, 
# it has a father-class named Operation. and i has following several class member variables :
# erive_action_captions: bool,derive_action_captions_summary: bool,num_words_action_captions_summary: int,
# min_num_words_action_captions_summary: int,
# derive_object_detections: bool,
# derive_object_detections_summary: bool,
#  num_words_object_detections_summary: int,
# min_num_words_object_detections_summary: int,
# derive_temporal_grounding: bool,
# derive_temporal_grounding_summary: bool,
# num_words_temporal_grounding_summary: int,
# min_num_words_temporal_grounding_summary: int,
# normalization_video_length: int = 180
#  this class has also two Function members: _execute and  get_num_words_for_summary；
# _execute has given inputs :(graph: Optional[Graph], api: Optional[API], target=Optional[Node]) return nothing
# it bases on the bool type varible, like derive_action_captions etc., to decide whether it calls 
# and runs input api to execute some inference like: action captions (using api can be LaViLa or CogAgent which are video-language understanding model ), 
# get unspecific objects(using api CogAgent) from given video clips and do temporal grounding(here using api also) for given video clip . 
# At the same time it decides whether do summary for these inference results respectively depending on the bool sign variables. 
# The number of words for summary depends on the output of function get_num_words_for_summary.  
# Finally respectively save all the results from above results in current node state in graph variables and output the each step in terminal using logger .
# get_num_words_for_summary this function is used for counting the number of words in summary, it has some inputs like :  
# len_video_sec,
# total_num_words_summary, 
# min_num_words_summary 
# function will output the number of words which is proportional to video length, 
# is a multiple of 10 and is no less than the minimum word requirement.

######################################################################################################################
######################################################################################################################
######################################################################################################################
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

class Operation:
    # Base class for operations (assuming it exists)
    pass

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
        self.derive_action_captions = derive_action_captions
        self.derive_action_captions_summary = derive_action_captions_summary
        self.num_words_action_captions_summary = num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary
        self.derive_object_detections = derive_object_detections
        self.derive_object_detections_summary = derive_object_detections_summary
        self.num_words_object_detections_summary = num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary
        self.derive_temporal_grounding = derive_temporal_grounding
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.num_words_temporal_grounding_summary = num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary
        self.normalization_video_length = normalization_video_length

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node'] = None
    ):
        """
        Executes the derivation and summarization processes based on the provided flags.
        """
        if graph is None or api is None or target is None:
            logger.error("Graph, API, or target node is None.")
            return

        # Assume target node has a method to get the video length in seconds
        len_video_sec = target.get_video_length()

        # Derive action captions
        if self.derive_action_captions:
            logger.info("Deriving action captions.")
            action_captions = api.derive_action_captions(target)
            target.state['action_captions'] = action_captions
            logger.info("Action captions derived.")

            # Summarize action captions if required
            if self.derive_action_captions_summary:
                logger.info("Summarizing action captions.")
                num_words = self.get_num_words_for_summary(
                    len_video_sec,
                    self.num_words_action_captions_summary,
                    self.min_num_words_action_captions_summary
                )
                summary = api.summarize_action_captions(action_captions, num_words)
                target.state['action_captions_summary'] = summary
                logger.info("Action captions summarized.")

        # Derive object detections
        if self.derive_object_detections:
            logger.info("Deriving object detections.")
            object_detections = api.derive_object_detections(target)
            target.state['object_detections'] = object_detections
            logger.info("Object detections derived.")

            # Summarize object detections if required
            if self.derive_object_detections_summary:
                logger.info("Summarizing object detections.")
                num_words = self.get_num_words_for_summary(
                    len_video_sec,
                    self.num_words_object_detections_summary,
                    self.min_num_words_object_detections_summary
                )
                summary = api.summarize_object_detections(object_detections, num_words)
                target.state['object_detections_summary'] = summary
                logger.info("Object detections summarized.")

        # Derive temporal grounding
        if self.derive_temporal_grounding:
            logger.info("Deriving temporal grounding.")
            temporal_grounding = api.derive_temporal_grounding(target)
            target.state['temporal_grounding'] = temporal_grounding
            logger.info("Temporal grounding derived.")

            # Summarize temporal grounding if required
            if self.derive_temporal_grounding_summary:
                logger.info("Summarizing temporal grounding.")
                num_words = self.get_num_words_for_summary(
                    len_video_sec,
                    self.num_words_temporal_grounding_summary,
                    self.min_num_words_temporal_grounding_summary
                )
                summary = api.summarize_temporal_grounding(temporal_grounding, num_words)
                target.state['temporal_grounding_summary'] = summary
                logger.info("Temporal grounding summarized.")

    def get_num_words_for_summary(
        self,
        len_video_sec: int,
        total_num_words_summary: int,
        min_num_words_summary: int
    ) -> int:
        """
        Calculates the number of words for the summary based on video length.
        Ensures the number is proportional, a multiple of 10, and not less than the minimum required.
        """
        num_words = int((len_video_sec / self.normalization_video_length) * total_num_words_summary)
        num_words = max(num_words, min_num_words_summary)
        # Round up to the nearest multiple of 10
        num_words = int(math.ceil(num_words / 10.0)) * 10
        return num_words


######################################################################################################################
######################################################################################################################
######################################################################################################################

# QUESTIONS (given context code)：

# Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. Please do not start when i still don't show you my code.

# it has a father-class named Operation. and i has following several class member variables :
# erive_action_captions: bool,derive_action_captions_summary: bool,num_words_action_captions_summary: int,
# min_num_words_action_captions_summary: int,
# derive_object_detections: bool,
# derive_object_detections_summary: bool,
#  num_words_object_detections_summary: int,
# min_num_words_object_detections_summary: int,
# derive_temporal_grounding: bool,
# derive_temporal_grounding_summary: bool,
# num_words_temporal_grounding_summary: int,
# min_num_words_temporal_grounding_summary: int,
# normalization_video_length: int = 180
#  this class has also two Function members: _execute and  get_num_words_for_summary；
# _execute has given inputs :(graph: Optional[Graph], api: Optional[API], target=Optional[Node]) return nothing
# it bases on the bool type varible, like derive_action_captions etc., 
# to decide whether it calls and runs input api to execute some inference like: 
# action captions (using api can be LaViLa or CogAgent which are video-language understanding model ), 
# get unspecific objects(using api CogAgent) from given video clips and do temporal grounding(here using api also) for given video clip . 
# At the same time it decides whether do summary for these inference results respectively depending on the bool sign variables. 
# The number of words for summary depends on the output of function get_num_words_for_summary.  
# Finally respectively save all the results from above results in current node state in graph variables and output the each step in terminal using logger .
# get_num_words_for_summary this function is used for counting the number of words in summary, it has some inputs like :  
# len_video_sec,
# total_num_words_summary, 
# min_num_words_summary 
# function will output the number of words which is proportional to video length, 
# is a multiple of 10 and is no less than the minimum word requirement.


# ANSWER:

# Please share your code whenever you're ready, and
# I’ll help extend it to meet the requirements you've outlined. 
# I'll wait for the code before proceeding.

# QUESTIONS:
# This is my partly finished class, please help me to finish it completely : showing the original code frame. 

# ANSWER:

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

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node] = None) -> None:
        root_node = graph.root
        len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

        if self.derive_action_captions:
            # Derive action captions using the API
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            # Summarize action captions
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=root_node.state.spatial_node_state.action_captions,
                object_detections=[],
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
            )

            logger.info(f"Derived action caption summary for root node {root_node}.")
            logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

        if self.derive_object_detections:
            # Derive object detections using the API
            root_node.state.spatial_node_state.object_detections = api.get_object_detections_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            # Summarize object detections
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=root_node.state.spatial_node_state.object_detections,
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
            )

            logger.info(f"Derived object detection summary for root node {root_node}.")
            logger.debug(f"Object Detections Summary:\n{root_node.state.spatial_node_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            # Derive temporal grounding using the API
            root_node.state.spatial_node_state.temporal_grounding = api.get_temporal_grounding_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived temporal grounding for root node {root_node}.")
            logger.debug(f"Temporal Grounding:\n{root_node.state.spatial_node_state.temporal_grounding}")

        if self.derive_temporal_grounding_summary:
            # Summarize temporal grounding
            root_node.state.spatial_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=[],
                temporal_grounding=root_node.state.spatial_node_state.temporal_grounding,
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
            )

            logger.info(f"Derived temporal grounding summary for root node {root_node}.")
            logger.debug(f"Temporal Grounding Summary:\n{root_node.state.spatial_node_state.temporal_grounding_summary}")

    def get_num_words_for_summary(self, len_video_sec: int, total_num_words_summary: int, min_num_words_summary: int) -> int:
        """
        Calculates the number of words for the summary based on video length.
        Ensures the number is proportional, a multiple of 10, and not less than the minimum required.
        """
        num_words = int((len_video_sec / self.normalization_video_length) * total_num_words_summary)
        num_words = max(num_words, min_num_words_summary)
        # Round up to the nearest multiple of 10
        num_words = ((num_words + 9) // 10) * 10
        return num_words


# QUESTION (BLANKS COMPLETION)

# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.

# the class framework called DeriveRootNodeState, 
# it has a father-class named Operation. and i has following several class member variables :
# erive_action_captions: bool,derive_action_captions_summary: bool,num_words_action_captions_summary: int,
# min_num_words_action_captions_summary: int,
# derive_object_detections: bool,
# derive_object_detections_summary: bool,
#  num_words_object_detections_summary: int,
# min_num_words_object_detections_summary: int,
# derive_temporal_grounding: bool,
# derive_temporal_grounding_summary: bool,
# num_words_temporal_grounding_summary: int,
# min_num_words_temporal_grounding_summary: int,
# normalization_video_length: int = 180
# this class has also two Function members: _execute and  get_num_words_for_summary；
# _execute has given inputs :(graph: Optional[Graph], api: Optional[API], target=Optional[Node]) return nothing
# it bases on the bool type varible, like derive_action_captions etc., to decide whether it calls and
# runs input api to execute some inference like: action captions (using api can be LaViLa or CogAgent which are video-language understanding model ),
# get unspecific objects(using api CogAgent) from given video clips and do temporal grounding(here using api also) for given video clip .
# At the same time it decides whether do summary for these inference results respectively depending on the bool sign variables.
# The number of words for summary depends on the output of function get_num_words_for_summary.
# Finally respectively save all the results from above results in current node state in graph variables and output the each step in terminal using logger .
# get_num_words_for_summary this function is used for counting the number of words in summary, it has some inputs like :  
# len_video_sec,
# total_num_words_summary, 
# min_num_words_summary 
# function will output the number of words which is proportional to video length, 
# is a multiple of 10 and is no less than the minimum word requirement.

# ANSWERS

# Understood! Once you share your incomplete class code,
# I'll help fill in the gaps to complete it. 
# Let me know when you're ready for me to start.


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

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node] = None) -> None:
        root_node = graph.root
        len_video_sec = len(root_node.state.video_clip) / root_node.state.video_clip.sampled_fps

        if self.derive_action_captions:
            # Get action captions from the video clip
            root_node.state.spatial_node_state.action_captions = api.get_action_captions_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived action captions for root node {root_node}.")
            logger.debug(f"Action Captions:\n{root_node.state.spatial_node_state.action_captions}")

        if self.derive_action_captions_summary:
            # Summarize action captions
            root_node.state.spatial_node_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=root_node.state.spatial_node_state.action_captions,
                object_detections=[],
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
            )

            logger.info(f"Derived action caption summary for root node {root_node}.")
            logger.debug(f"Action Caption Summary:\n{root_node.state.spatial_node_state.action_captions_summary}")

        if self.derive_object_detections:
            # Get object detections from the video clip
            root_node.state.spatial_node_state.object_detections = api.get_unspecific_objects_from_video_clip(
                video_clip=root_node.state.video_clip
            )

            logger.info(f"Derived object detections for root node {root_node}.")
            logger.debug(f"Object Detections:\n{root_node.state.spatial_node_state.object_detections}")

        if self.derive_object_detections_summary:
            # Summarize object detections
            root_node.state.spatial_node_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=root_node.state.spatial_node_state.get_textual_object_list(),
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
            )

            logger.info(f"Derived object detection summary for root node {root_node}.")
            logger.debug(f"Object Detection Summary:\n{root_node.state.spatial_node_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            # Get temporal grounding from the video clip and the question
            temporal_grounding = api.get_temporal_grounding_from_video_clip_and_text(
                video_clip=root_node.state.video_clip,
                text=root_node.state.task.question
            )

            root_node.state.temporal_node_state.salience_indicators = temporal_grounding["salience_indicators"]

            logger.info(f"Derived temporal grounding for root node {root_node}.")
            logger.debug(f"Temporal Grounding:\n{temporal_grounding}")

        if self.derive_temporal_grounding_summary:
            # Summarize temporal grounding
            root_node.state.temporal_node_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=[],
                temporal_grounding=root_node.state.temporal_node_state.salience_indicators,
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_node.state.video_clip,
                question=root_node.state.task.question,
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
            )

            logger.info(f"Derived temporal grounding summary for root node {root_node}.")
            logger.debug(f"Temporal Grounding Summary:\n{root_node.state.temporal_node_state.temporal_grounding_summary}")

        root_node.state.derived = True

        logger.info(f"Executed root state derivation operation: DeriveUniversalState")
        logger.debug(f"The following universal node state has been derived:\n{root_node.state}")

    def get_num_words_for_summary(
            self,
            len_video_sec,
            total_num_words_summary,
            min_num_words_summary
    ):
        # Do not normalize if normalization_video_length is -1
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
        else:
            # Calculate proportional number of words based on video length
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
        # Round to nearest multiple of 10
        num_words_mul_ten = int(round(whole_video_word_contingent / 10)) * 10
        # Ensure minimum word count is met
        num_words = max(num_words_mul_ten, min_num_words_summary)
        return num_words
    

    
