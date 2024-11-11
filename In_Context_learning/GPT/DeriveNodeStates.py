 # GPT-o1
 # Ground-Truth
# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# You can imitate and learn this ability from the example that i will show you

# the example for requirement and correspongding code:

# requirement:
# a class called DeriveRootNodeState, 
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
# this class has also two Function members: _execute and get_num_words_for_summary；
# _execute has given inputs :(graph: Optional[Graph], api: Optional[API], target=Optional[Node]) return nothing
# first, it take root node from graph and set it to root_node, compute the len of current video clip
# video clip is also from this form 'root_node.state.video_clip'
# node this variable has multi-sublevel, using '.' to call ,the form is like root_node.state.spatial_node_state.action_captions 
# if self.derive_action_aption is true , it will use api's function to get action captions from clips 
# and output the info and debug in terminal
# next if self.derive_action_captions_summary is true, it uses api's func get summry_from_noisy_perceptive data() to 
# get summary from above step (action captions) 
# this function has follwing inputs:action_captions ;object_detections
# ;temporal_grounding;interleaved_action_captions_and_object_detections
# video_clip;question;words. word is the result from another member function.
# These 7 input parameters are needed to be assgined with appropriate value or variables.
# same as the step in actions caption still need to logger info and debug for summary.
# These 2 step will similarly be run for the get_unspecific_objects_from_video_clip ; get_summary_from_noisy_perceptive_data ;
# get_temporal_grounding_from_video_clip_and_text；get_summary_from_noisy_perceptive_data.
# additionally, the results from temporal_grounding has three parts :1.foreground_indicators;2.relevance_indicators3.salience_indicators.You need to respectively save them in node.state
# all of above generated results will be respectively saved into the root_node.state. And set the root_node.state.derived to True
# get_num_words_for_summary take len_video_sec;total_num_words_summary;min_num_words_summary as input and return num_words
# it does not normalize the number of words for the summary if the normalization video length is -1
# whole_video_word_contingent this variable is equal to total_num_words_summary
# otherwise it will normalize normalize the number of words,and calculate the number of words for the summary (proportionate to the video length in seconds)
# and it makes sure the num of words is multiple of 10 and no less than min_num_words_summary 
# code:


# please help me to finish it.
#######################################################################
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

        # Initialize action captions related members
        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        # Initialize object detections related members
        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        # Initialize temporal grounding related members
        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        # Initialize normalization video length
        self.normalization_video_length = normalization_video_length

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node'] = None
    ) -> None:
        if graph is None or api is None:
            logger.error("Graph or API is not provided.")
            return

        root_state = graph.root.state

        if not root_state.derived:
            logger.error("Root node state has not been derived yet.")
            return

        derivable_nodes = graph.get_derivable_nodes()

        for derivable_node in derivable_nodes:
            # Extract start and end frame indices
            total_start_frame_index = derivable_node.state.video_clip.sampled_indices[0]
            total_end_frame_index = derivable_node.state.video_clip.sampled_indices[-1]

            # Find list indices in root video's sampled indices
            try:
                start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
                end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
            except ValueError as e:
                logger.error(f"Frame index not found in root video sampled indices: {e}")
                continue

            # Calculate lengths in seconds
            len_clip_sec = len(derivable_node.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps

            logger.debug(
                f"Processing derivable node {derivable_node} with clip length {len_clip_sec} seconds "
                f"out of total video length {len_video_sec} seconds."
            )

            # Derive Action Captions
            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions

                logger.info(f"Derived action captions for derivable node {derivable_node}.")
                logger.debug(f"Action Captions for {derivable_node}: {clip_node_action_captions}")

            # Derive Action Captions Summary
            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
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

                logger.info(f"Derived action captions summary for derivable node {derivable_node}.")
                logger.debug(f"Action Captions Summary for {derivable_node}: {action_captions_summary}")

            # Derive Object Detections
            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[start_list_index:end_list_index + 1]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections

                logger.info(f"Derived object detections for derivable node {derivable_node}.")
                logger.debug(f"Object Detections for {derivable_node}: {clip_node_object_detections}")

            # Derive Object Detections Summary
            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
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

                logger.info(f"Derived object detections summary for derivable node {derivable_node}.")
                logger.debug(f"Object Detections Summary for {derivable_node}: {object_detections_summary}")

            # Derive Temporal Grounding
            if self.derive_temporal_grounding:
                derivable_node.state.temporal_node_state.foreground_indicators = root_state.temporal_node_state.foreground_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.relevance_indicators = root_state.temporal_node_state.relevance_indicators[start_list_index:end_list_index + 1]
                derivable_node.state.temporal_node_state.salience_indicators = root_state.temporal_node_state.salience_indicators[start_list_index:end_list_index + 1]

                logger.info(f"Derived temporal grounding for derivable node {derivable_node}.")
                logger.debug(
                    f"Temporal Grounding for {derivable_node}: "
                    f"{derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}"
                )

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
                    temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )

                derivable_node.state.temporal_node_state.temporal_grounding_summary = temporal_grounding_summary

                logger.info(f"Derived temporal grounding summary for derivable node {derivable_node}.")
                logger.debug(f"Temporal Grounding Summary for {derivable_node}: {temporal_grounding_summary}")

            # Mark the node as derived
            derivable_node.state.derived = True

            logger.info(f"Derived state for node {derivable_node}.")
            logger.debug(f"Derived Node State: {derivable_node.state}")

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
        else:
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec

        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        num_words_mul_ten = (int(round(num_words_exact / 10))) * 10
        num_words = max(num_words_mul_ten, min_num_words_summary)

        return num_words
    
#####################################################################################################3####################
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

        # Action Captions
        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        # Object Detections
        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        # Temporal Grounding
        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        # Normalization
        self.normalization_video_length = normalization_video_length

    def _execute(self, graph: Optional['Graph'], api: Optional['API'], target: Optional['Node']) -> None:
        if graph is None or api is None:
            err_msg = "Graph and API must be provided."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Get the root node state from the graph
        root_state = graph.root.state

        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Retrieve derivable nodes from the graph
        derivable_nodes = graph.get_derivable_nodes()
        logger.debug(f"Number of derivable nodes: {len(derivable_nodes)}")

        for derivable_node in derivable_nodes:
            logger.debug(f"Processing derivable node: {derivable_node}")

            # Find start and end indices of the video clip
            sampled_indices = derivable_node.state.video_clip.sampled_indices
            if not sampled_indices:
                logger.error(f"No sampled indices found for node {derivable_node}. Skipping.")
                continue  # Skip nodes without sampled indices

            total_start_frame_index = sampled_indices[0]
            total_end_frame_index = sampled_indices[-1]
            logger.debug(f"Total start frame index: {total_start_frame_index}")
            logger.debug(f"Total end frame index: {total_end_frame_index}")

            # Get the list indices in the root node's sampled indices
            try:
                start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
                end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
            except ValueError as e:
                logger.error(f"Frame index not found in root node's sampled indices: {e}")
                continue  # Skip nodes with invalid frame indices

            logger.debug(f"Start list index: {start_list_index}")
            logger.debug(f"End list index: {end_list_index}")

            # Calculate the length of the clip and the whole video in seconds
            len_clip_sec = len(derivable_nebaye.state.video_clip) / derivable_node.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logger.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logger.debug(f"Length of the whole video in seconds: {len_video_sec}")

            # Derive Action Captions
            if self.derive_action_captions:
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.action_captions = clip_node_action_captions

                logger.info(f"Inherited action captions from root node for node {derivable_node}.")
                logger.debug(f"Action Captions for node {derivable_node}: {clip_node_action_captions}")

            # Derive Action Captions Summary
            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
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
                logger.debug(f"Action Captions Summary for node {derivable_node}: {action_captions_summary}")

            # Derive Object Detections
            if self.derive_object_detections:
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                    start_list_index:end_list_index + 1
                ]
                derivable_node.state.spatial_node_state.object_detections = clip_node_object_detections

                logger.info(f"Inherited object detections from root node for node {derivable_node}.")
                logger.debug(f"Object Detections for node {derivable_node}: {clip_node_object_detections}")

            # Derive Object Detections Summary
            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
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
                logger.debug(f"Object Detections Summary for node {derivable_node}: {object_detections_summary}")

            # Derive Temporal Grounding
            if self.derive_temporal_grounding:
                temporal_grounding = root_state.temporal_node_state.get_temporal_grounding_subset(
                    start_index=start_list_index,
                    end_index=end_list_index
                )
                derivable_node.state.temporal_node_state.foreground_indicators = temporal_grounding["foreground_indicators"]
                derivable_node.state.temporal_node_state.relevance_indicators = temporal_grounding["relevance_indicators"]
                derivable_node.state.temporal_node_state.salience_indicators = temporal_grounding["salience_indicators"]

                logger.info(f"Inherited temporal grounding from root node for node {derivable_node}.")
                logger.debug(f"Temporal Grounding for node {derivable_node}: {derivable_node.state.temporal_node_state.get_textual_temporal_grounding()}")

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
                    temporal_grounding=[derivable_node.state.temporal_node_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    options=derivable_node.state.task.options,
                    words=num_words
                )
                derivable_node.state.temporal_node_state.temporal_grounding_summary = temporal_grounding_summary

                logger.info(f"Derived temporal grounding summary for node {derivable_node}.")
                logger.debug(f"Temporal Grounding Summary for node {derivable_node}: {temporal_grounding_summary}")

            # Mark the node as derived
            derivable_node.state.derived = True
            logger.info(f"Derived node state for node {derivable_node}.")
            logger.debug(f"Derived node state details:\n{derivable_node.state}")

        logger.info("Executed state derivation operation: DeriveNodeStatesOfDerivableNodes")

    def get_num_words_for_summary(
            self,
            len_clip_sec: float,
            len_video_sec: float,
            total_num_words_summary: int,
            min_num_words_summary: int
    ) -> int:
        # Do not normalize the number of words for the summary if the normalization video length is -1
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
            logger.debug("Normalization is disabled. Using total_num_words_summary as is.")
        else:
            # Calculate the word contingent for the whole video
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
            logger.debug(f"Whole video word contingent (before normalization): {whole_video_word_contingent}")

        # Calculate the exact number of words for the clip
        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        logger.debug(f"Exact number of words for the clip: {num_words_exact}")

        # Make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(num_words_exact / 10))) * 10
        logger.debug(f"Number of words rounded to multiple of 10: {num_words_mul_ten}")

        # Ensure the number of words is at least the minimum required
        num_words = max(num_words_mul_ten, min_num_words_summary)
        logger.debug(f"Final number of words for summary (max with min_num_words_summary): {num_words}")

        return num_words
#######################################################################################################################
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

        # Initialize normalization video length
        self.normalization_video_length = normalization_video_length

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node] = None
    ) -> None:
        """
        Executes the state derivation operation on derivable nodes within the graph.

        Parameters:
            graph (Optional[Graph]): The graph containing nodes.
            api (Optional[API]): The API instance for summarization.
            target (Optional[Node]): The target node (not used in this implementation).

        Returns:
            None
        """
        if graph is None or api is None:
            logger.error("Graph and API must not be None.")
            raise ValueError("Graph and API must not be None.")

        # Get the root node state from the graph
        root_state = graph.root.state

        # Check if the root state has been derived
        if not root_state.derived:
            err_msg = "Root node state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Get derivable nodes from the graph
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
                # Inherit the action captions from the root node
                clip_node_action_captions = root_state.spatial_node_state.action_captions[
                    start_list_index:end_list_index + 1
                ]

                # Set the action captions of the derivable node
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

                # Use action caption summary API function to summarize all action captions to get a clip summary
                action_caption_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_node.state.spatial_node_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )

                # Set the action captions summary of the derivable node
                derivable_node.state.spatial_node_state.action_captions_summary = action_caption_summary

                logger.info(f"Derived action caption summary for node {derivable_node}.")
                logger.debug(f"Action Caption Summary:\n{derivable_node.state.spatial_node_state.action_captions_summary}")

            # Derive Object Detections
            if self.derive_object_detections:
                # Inherit the object detections from the root node
                clip_node_object_detections = root_state.spatial_node_state.object_detections[
                    start_list_index:end_list_index + 1
                ]

                # Set the object detections of the derivable node
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

                # Use object detection summary API function to summarize all object detections to get a universal video summary
                object_detection_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_node.state.spatial_node_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_node.state.video_clip,
                    question=derivable_node.state.task.question,
                    words=num_words
                )

                # Set the object detections summary of the derivable node
                derivable_node.state.spatial_node_state.object_detections_summary = object_detection_summary

                logger.info(f"Derived object detection summary for node {derivable_node}.")
                logger.debug(f"Object Detection Summary:\n{derivable_node.state.spatial_node_state.object_detections_summary}")

            # Derive Temporal Grounding
            if self.derive_temporal_grounding:
                # Inherit the temporal grounding from the root node
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

                # Use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
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

                # Set the temporal grounding summary of the derivable node
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
        Calculates the number of words to be used for summarizing based on the length of the clip and the entire video.

        Parameters:
            len_clip_sec (float): Length of the clip in seconds.
            len_video_sec (float): Length of the entire video in seconds.
            total_num_words_summary (int): Total number of words allocated for the summary.
            min_num_words_summary (int): Minimum number of words required for the summary.

        Returns:
            int: The calculated number of words for the summary.
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