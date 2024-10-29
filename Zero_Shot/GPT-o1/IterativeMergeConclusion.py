# Ground Truth
class IterativeMergeConclusion(Operation):
    def __init__(
            self,
            qa_prompt: str,
            qa_completion_start: str,
            qa_max_new_tokens: int,
            qa_temperature: float,
            qa_replace_c: bool,
            qa_parse_strategy: str,
            self_reflect_prompt: str,
            self_reflect_completion_start: str,
            self_reflect_max_new_tokens: int,
            self_reflect_temperature: float,
            self_reflect_parse_strategy: str
    ):
        super().__init__()

        self.qa_prompt = qa_prompt
        self.qa_completion_start = qa_completion_start
        self.qa_max_new_tokens = qa_max_new_tokens
        self.qa_temperature = qa_temperature
        self.qa_replace_c = qa_replace_c
        self.qa_parse_strategy = qa_parse_strategy

        self.self_reflect_prompt = self_reflect_prompt
        self.self_reflect_completion_start = self_reflect_completion_start
        self.self_reflect_max_new_tokens = self_reflect_max_new_tokens
        self.self_reflect_temperature = self_reflect_temperature
        self.self_reflect_parse_strategy = self_reflect_parse_strategy

    def _execute(self, graph: Optional[Graph], api: Optional[API], target=Optional[Node]) -> dict[str, any]:
        answerable_nodes = [node for node in graph.get_concludable_nodes() if node.state.ranking_confidence is not None]
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        # shallow copy the list of answerable nodes but use a new list
        answerable_nodes = list(answerable_nodes)

        # now we can change the order of the new list without affecting the order of the original list
        # specifically, we want to sort the nodes by their ranking confidence (from highest to lowest)
        answerable_nodes.sort(key=lambda x: x.state.ranking_confidence, reverse=True)

        # get the most answerable node with the center point of its interval
        answerable_concatenated_summaries = {}

        # remember the inference data
        conclusions = {}

        # iterate over the nodes and merge with the next node in the sorted list
        for i, node in enumerate(answerable_nodes):

            # get the center point of the interval of the node
            interval_start = node.state.video_clip.sampled_indices[0]
            interval_end = node.state.video_clip.sampled_indices[-1]
            interval_center = (interval_start + interval_end) / 2

            # add the node to the answerable_concatenated_summaries
            answerable_concatenated_summaries[interval_center] = node.state.get_lexical_representation()

            # experimentally tested with llama 3 that there is almost 3% performance increase when collecting
            # all nodes with level 3 ranking_confidence before doing the first merge and inference
            if i != len(answerable_nodes) - 1:
                # continue as long as there is a next node that also has a ranking_confidence of 3
                if answerable_nodes[i + 1].state.ranking_confidence == 3:
                    continue
                # if the next node does not have a ranking_confidence of 3, merge the current nodes for the first time

            # If the first node has no ranking_confidence of 3, we will skip to the next node.
            # This is because we want to merge if the first node has not the highest possible confidence.
            # In case it has the highest confidence, we will use it alone for the final prediction.
            # if i == 0 and node.state.ranking_confidence != 3:
            #     continue

            # otherwise we will merge the lexical states of all current nodes in their temporal order
            # sort the answerable_concatenated_summaries by the center of the interval
            answerable_concatenated_summaries = dict(sorted(answerable_concatenated_summaries.items()))

            # concatenate the lexical node states
            concatenated_lexical_state = "\n".join(answerable_concatenated_summaries.values())

            # note that our framework currently supports free-form questions and single-choice questions
            # decide which candidate derivation function to use based on availability of the task options
            # check if all options are "N/A", i.e. not available
            if all([option == "N/A" for option in task.options.values()]):
                # get the free-form QA prediction
                intermediate_candidate = derive_free_form_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_lexical_state,
                    question=task.question,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = intermediate_candidate[0]
            else:
                # get the single-choice QA prediction
                intermediate_candidate = derive_options_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_lexical_state,
                    task=task,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = intermediate_candidate[0][0] if intermediate_candidate[
                    0] else "failed to parse prediction from completion"

            # remember the whole video state, the completion and the QA prediction
            conclusion = {
                "concatenated_lexical_state": concatenated_lexical_state,
                "completion": intermediate_candidate[1],
                "final_ranking": intermediate_candidate[0],
                "final_prediction": final_prediction
            }

            logger.debug(f"Whole video state (i={i}): {answerable_concatenated_summaries}")
            logger.debug(f"Completion (i={i}): {intermediate_candidate[1]}")
            logger.debug(f"Predicted answer (i={i}): {intermediate_candidate[0]}")

            # remember the conclusion together with the center of the interval of the new node
            conclusions[i] = {
                "node_interval_center": interval_center,
                "conclusion": conclusion
            }

            # evaluate answerability of the intermediate prediction like in Video Agent
            # compare https://arxiv.org/abs/2403.11481

            # combine the QA prompt with the completion to get the reasoning history
            reasoning_history = f"### Exam Task\n\n{intermediate_candidate[2]}### Student Answer\n\n{intermediate_candidate[1]}"

            # there must nothing be parsed from the reasoning history
            reasoning_history = reasoning_history.replace("{", "{{").replace("}", "}}")

            # build the self-reflection prompt for the self-reflection
            self_reflection_prompt = self.self_reflect_prompt.format(
                reasoning_history=reasoning_history
            )

            # get the completion of the self-reflection
            completion = api.get_completion_from_text(
                text=self_reflection_prompt,
                completion_start=self.self_reflect_completion_start,
                max_new_tokens=self.self_reflect_max_new_tokens,
                temperature=self.self_reflect_temperature
            )
            logger.debug(f"Derived llm completion of self-reflected confidence (i={i}): {completion}")

            # parse the answerability from the completion
            # define the priority of certain keywords
            keywords_in_priority_order = [
                'confidence',
                'level',
                'conf',
                'answerability',
                'confidence_level',
                'confidence level'
            ]
            candidate = parse_answer_json(
                text=completion,
                keywords_in_priority_order=keywords_in_priority_order,
                candidate_fun=get_single_number_candidates_from_text
            )

            # assure valid numerical confidence
            confidence = 1 if not candidate else candidate
            confidence = max(1, min(3, confidence))
            logger.debug(f"Parsed self-reflected confidence from completion (i={i}): {confidence}")

            # if the confidence is 3, we can stop the iteration and use the last QA prediction
            if confidence == 3:
                break

        # take last intermediate prediction as final prediction
        conclusion = {
            "number_of_iterations": len(conclusions),
            "iteration_conclusions": conclusions,
            "whole_video_state": answerable_concatenated_summaries,
            "completion": list(conclusions.values())[-1]["conclusion"]["completion"],
            "final_ranking": list(conclusions.values())[-1]["conclusion"]["final_ranking"],
            "final_prediction": list(conclusions.values())[-1]["conclusion"]["final_prediction"],
        }

        logger.info(f"Executed llm-based conclusion operation: "
                    f"IterativeMergeConclusion -> {conclusion['final_prediction']}")

        return conclusion
##############################################################################################
##############################################################################################3
# Hi，you are an very experienced programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially for using Python completely to satisfy the requirement.
# Next please complete the code, including classes and functions for my algorithm framework under my instruction and decription

# Please help me to write a class named IterativeMergeConclusion， it takes Operation as baseclass. and its inputs for the class are  qa_prompt: str, qa_completion_start: str,qa_max_new_tokens: int, qa_temperature: float, qa_replace_c: bool,qa_parse_strategy: str,self_reflect_prompt: str,self_reflect_completion_start: str,self_reflect_max_new_tokens: int,self_reflect_temperature: float,self_reflect_parse_strategy: str
# and of course there are some member variables to be initialized: self.qa_prompt = qa_prompt, self.qa_completion_start = qa_completion_start,self.qa_max_new_tokens = qa_max_new_tokens, self.qa_temperature = qa_temperature,self.qa_replace_c = qa_replace_c,self.qa_parse_strategy = qa_parse_strategy,self.self_reflect_prompt = self_reflect_prompt, self.self_reflect_completion_start = self_reflect_completion_start,self.self_reflect_max_new_tokens = self_reflect_max_new_tokens,self.self_reflect_temperature = self_reflect_temperature, self.self_reflect_parse_strategy = self_reflect_parse_strategy
# the function is named _execute, which has inputs including graph: Optional[Graph], api: Optional[API], target=Optional[Node] and it returns dict[str, any]
# firstly get all answerable_nodes via graph.get_concludable_nodes(), and if the node.state.ranking_confidence is not none, you can add this node to the answerable_nodes list
# then use graph.root.state.spatial_node_state.action_captions_summary to return whole_video_summary and use graph.root.state.task to return variable task.
# then use list[] to get shallow copy of answerable_nodes. and next .sort() to sort answerable_nodes depending the nodes.state.ranking_confidence in list from highest to lowest.
# initialize the answerable_concatenated_summaries with {} and following see the center point of interval is the most answerable node
# initialize the conclusions {}
# iterate over answerable_nodes using enumerate and get the center point of the interval of videoclip presented by node
# interval_start from node.state.video_clip.sampled_indices[0]; end from node.state.video_clip.sampled_indices[-1]. And after adding them divide it by 2
# add the node to the answerable_concatenated_summaries,corresponding key is interval_center ,value is node.state.get_lexical_representation()
# next using if i current index for enumerate is not the last index and next element(node) in answerable_nodes 's state.ranking_confidence is 3 then continue this loop
# then next dict（sorted) the answerable_concatenated_summaries and join "\n" in answerable_concatenated_summaries.values()
# if all option in the task.options.values() are all "N/A" then all the function and returns the result as intermediate_candidate and set intermediate_candidate[0] to final_prediction
# otherwise call derive_options_candidate_from_whole_video_state to get the result as intermediate_candidate and then if intermediate_candidate[0] exists and set final_prediction as intermediate_candidate[0][0], otherwise set it to "failed to parse prediction from completion"
# create a dictionary named conclusion including "concatenated_lexical_state" "concatenated_lexical_state" ,"completion": intermediate_candidate[1],"final_ranking": intermediate_candidate[0],"final_prediction": final_prediction
# and then do logger.debug "Whole video state (i={i}): {answerable_concatenated_summaries}" ,"Completion (i={i}): {intermediate_candidate[1]}","Predicted answer (i={i}): {intermediate_candidate[0]}"
# and then set the conclusions[i] as "node_interval_center": interval_center,"conclusion": conclusion.
# create variable reaning_history = f"### Exam Task\n\n{intermediate_candidate[2]}### Student Answer\n\n{intermediate_candidate[1]}"
# and then use .replace() to replace "{" with "{{" and replace "}" with "}}".
# next do format operation to self.self_reflect_prompt to replace the reasoning_history in prompt with just gotten reasoning_history
# next call api.get_completion_from_text with the one of inputs(self_reflection_prompt) to get complection of self-reflection
# define the priority of certain keywords as dictionary keywords_in_priority_order likes 'confidence','level','conf','answerability','confidence_level','confidence level'
# then call parse_answer_json() using text=completion,keywords_in_priority_order=keywords_in_priority_order,candidate_fun=get_single_number_candidates_from_text as inputs to return candidate
# can then this candidate is not true or not existing then set confidence as 1 othterwise set confidence as candidate
# then make sure the confidence in range of 1-3 and if here confidence is equel to 3 then break stoping the iteration.
# finally generate a dictionary named conclusion including "number_of_iterations": len(conclusions),"iteration_conclusions": conclusions,"whole_video_state": answerable_concatenated_summaries,"completion": list(conclusions.values())[-1]["conclusion"]["completion"],"final_ranking": list(conclusions.values())[-1]["conclusion"]["final_ranking"],"final_prediction": list(conclusions.values())[-1]["conclusion"]["final_prediction"]
# this whole function finally returns the conclusion dictionary

################################################################################
class IterativeMergeConclusion(Operation):
    def __init__(
        self,
        qa_prompt: str,
        qa_completion_start: str,
        qa_max_new_tokens: int,
        qa_temperature: float,
        qa_replace_c: bool,
        qa_parse_strategy: str,
        self_reflect_prompt: str,
        self_reflect_completion_start: str,
        self_reflect_max_new_tokens: int,
        self_reflect_temperature: float,
        self_reflect_parse_strategy: str
    ):
        super().__init__()  # Initialize the base Operation class if necessary
        # Initialize member variables
        self.qa_prompt = qa_prompt
        self.qa_completion_start = qa_completion_start
        self.qa_max_new_tokens = qa_max_new_tokens
        self.qa_temperature = qa_temperature
        self.qa_replace_c = qa_replace_c
        self.qa_parse_strategy = qa_parse_strategy
        self.self_reflect_prompt = self_reflect_prompt
        self.self_reflect_completion_start = self_reflect_completion_start
        self.self_reflect_max_new_tokens = self_reflect_max_new_tokens
        self.self_reflect_temperature = self_reflect_temperature
        self.self_reflect_parse_strategy = self_reflect_parse_strategy

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node']
    ) -> Dict[str, Any]:
        # Step 1: Get all answerable nodes
        answerable_nodes: List['Node'] = []
        for node in graph.get_concludable_nodes():
            if node.state.ranking_confidence is not None:
                answerable_nodes.append(node)
        
        # Step 2: Retrieve whole video summary and task
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        # Step 3: Create a shallow copy and sort the answerable nodes
        sorted_answerable_nodes = list(answerable_nodes)
        sorted_answerable_nodes.sort(
            key=lambda x: x.state.ranking_confidence,
            reverse=True
        )

        # Step 4: Initialize summaries and conclusions
        answerable_concatenated_summaries: Dict[float, str] = {}
        conclusions: Dict[int, Dict[str, Any]] = {}

        # Step 5: Iterate over answerable nodes
        for i, node in enumerate(sorted_answerable_nodes):
            # Calculate interval center
            interval_start = node.state.video_clip.sampled_indices[0]
            interval_end = node.state.video_clip.sampled_indices[-1]
            interval_center = (interval_start + interval_end) / 2

            # Add node's lexical representation to summaries
            lexical_representation = node.state.get_lexical_representation()
            answerable_concatenated_summaries[interval_center] = lexical_representation

            # Check if next node has ranking_confidence of 3
            if i < len(sorted_answerable_nodes) - 1:
                next_node = sorted_answerable_nodes[i + 1]
                if next_node.state.ranking_confidence == 3:
                    logger.debug(
                        f"Skipping node at index {i} due to next node's ranking_confidence == 3."
                    )
                    continue

            # Concatenate summaries
            concatenated_sorted = "\n".join(answerable_concatenated_summaries.values())

            # Determine intermediate_candidate and final_prediction
            if all(option == "N/A" for option in task.options.values()):
                intermediate_candidate = {
                    '0': "all_options_not_available"
                }
                final_prediction = intermediate_candidate['0']
            else:
                # Assume derive_options_candidate_from_whole_video_state returns a tuple
                intermediate_candidate = derive_options_candidate_from_whole_video_state(
                    whole_video_summary, task
                )
                if intermediate_candidate and intermediate_candidate[0]:
                    final_prediction = intermediate_candidate[0][0]
                else:
                    final_prediction = "failed to parse prediction from completion"

            # Create conclusion dictionary
            conclusion = {
                "concatenated_lexical_state": concatenated_sorted,
                "completion": intermediate_candidate[1] if len(intermediate_candidate) > 1 else None,
                "final_ranking": intermediate_candidate[0] if len(intermediate_candidate) > 0 else None,
                "final_prediction": final_prediction
            }

            # Logging
            logger.debug(
                f"Whole video state (i={i}): {answerable_concatenated_summaries}"
            )
            logger.debug(
                f"Completion (i={i}): {intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            )
            logger.debug(
                f"Predicted answer (i={i}): {intermediate_candidate[0] if len(intermediate_candidate) > 0 else 'N/A'}"
            )

            # Add to conclusions
            conclusions[i] = {
                "node_interval_center": interval_center,
                "conclusion": conclusion
            }

            # Create reasoning history
            reasoning_history = f"### Exam Task\n\n{intermediate_candidate[2] if len(intermediate_candidate) > 2 else 'N/A'}### Student Answer\n\n{intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            # Replace curly braces
            reasoning_history = reasoning_history.replace("{", "{{").replace("}", "}}")
            # Format the self_reflect_prompt
            formatted_prompt = self.self_reflect_prompt.format(reasoning_history=reasoning_history)

            # Call API to get self-reflection completion
            completion = api.get_completion_from_text(
                prompt=formatted_prompt,
                start=self.self_reflect_completion_start,
                max_new_tokens=self.self_reflect_max_new_tokens,
                temperature=self.self_reflect_temperature
            )

            # Define keywords priority
            keywords_in_priority_order = {
                'confidence': 1,
                'level': 2,
                'conf': 3,
                'answerability': 4,
                'confidence_level': 5,
                'confidence level': 6
            }

            # Parse the answer JSON
            candidate = parse_answer_json(
                text=completion,
                keywords_in_priority_order=keywords_in_priority_order,
                candidate_fun=get_single_number_candidates_from_text
            )

            # Determine confidence
            if not candidate:
                confidence = 1
            else:
                confidence = candidate

            # Ensure confidence is within range 1-3
            confidence = max(1, min(confidence, 3))

            logger.debug(f"Confidence after parsing: {confidence}")

            # If confidence is 3, stop the iteration
            if confidence == 3:
                logger.debug("Confidence reached 3. Stopping iteration.")
                break

        # Generate the final conclusion dictionary
        if conclusions:
            last_conclusion = list(conclusions.values())[-1]["conclusion"]
            final_conclusion = {
                "number_of_iterations": len(conclusions),
                "iteration_conclusions": conclusions,
                "whole_video_state": answerable_concatenated_summaries,
                "completion": last_conclusion.get("completion"),
                "final_ranking": last_conclusion.get("final_ranking"),
                "final_prediction": last_conclusion.get("final_prediction")
            }
        else:
            final_conclusion = {
                "number_of_iterations": 0,
                "iteration_conclusions": {},
                "whole_video_state": {},
                "completion": None,
                "final_ranking": None,
                "final_prediction": None
            }

        return final_conclusion
    ####################################################################################
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description.  
# please do not start writing, when i have not given you the partly code.
##########################################################################################################33
class IterativeMergeConclusion(Operation):
    def __init__(
        self,
        qa_prompt: str,
        qa_completion_start: str,
        qa_max_new_tokens: int,
        qa_temperature: float,
        qa_replace_c: bool,
        qa_parse_strategy: str,
        self_reflect_prompt: str,
        self_reflect_completion_start: str,
        self_reflect_max_new_tokens: int,
        self_reflect_temperature: float,
        self_reflect_parse_strategy: str
    ):
        super().__init__()

        # Initialize QA member variables
        self.qa_prompt = qa_prompt
        self.qa_completion_start = qa_completion_start
        self.qa_max_new_tokens = qa_max_new_tokens
        self.qa_temperature = qa_temperature
        self.qa_replace_c = qa_replace_c
        self.qa_parse_strategy = qa_parse_strategy

        # Initialize self-reflection member variables
        self.self_reflect_prompt = self_reflect_prompt
        self.self_reflect_completion_start = self_reflect_completion_start
        self.self_reflect_max_new_tokens = self_reflect_max_new_tokens
        self.self_reflect_temperature = self_reflect_temperature
        self.self_reflect_parse_strategy = self_reflect_parse_strategy

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node']
    ) -> Dict[str, Any]:
        # Step 1: Retrieve all answerable nodes with non-None ranking_confidence
        answerable_nodes: List['Node'] = [
            node for node in graph.get_concludable_nodes()
            if node.state.ranking_confidence is not None
        ]

        # Step 2: Extract whole video summary and task from the graph
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        # Step 3: Create a shallow copy and sort the answerable nodes by ranking_confidence (descending)
        sorted_answerable_nodes = list(answerable_nodes)
        sorted_answerable_nodes.sort(
            key=lambda x: x.state.ranking_confidence,
            reverse=True
        )

        # Step 4: Initialize summaries and conclusions
        answerable_concatenated_summaries: Dict[float, str] = {}
        conclusions: Dict[int, Dict[str, Any]] = {}

        # Step 5: Iterate over the sorted answerable nodes
        for i, node in enumerate(sorted_answerable_nodes):
            # Calculate the center point of the video clip interval
            interval_start = node.state.video_clip.sampled_indices[0]
            interval_end = node.state.video_clip.sampled_indices[-1]
            interval_center = (interval_start + interval_end) / 2

            # Add the node's lexical representation to the summaries
            lexical_representation = node.state.get_lexical_representation()
            answerable_concatenated_summaries[interval_center] = lexical_representation

            # Check if the next node has a ranking_confidence of 3 to possibly skip
            if i < len(sorted_answerable_nodes) - 1:
                next_node = sorted_answerable_nodes[i + 1]
                if next_node.state.ranking_confidence == 3:
                    logger.debug(
                        f"Skipping node at index {i} due to next node's ranking_confidence == 3."
                    )
                    continue

            # Concatenate all lexical representations separated by newlines
            concatenated_sorted = "\n".join(answerable_concatenated_summaries.values())

            # Determine if all task options are "N/A"
            if all(option == "N/A" for option in task.options.values()):
                # Get the free-form QA prediction
                intermediate_candidate = derive_free_form_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_sorted,
                    question=task.question,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = intermediate_candidate[0]
            else:
                # Get the single-choice QA prediction
                intermediate_candidate = derive_options_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_sorted,
                    task=task,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = (
                    intermediate_candidate[0][0]
                    if intermediate_candidate[0]
                    else "failed to parse prediction from completion"
                )

            # Create the conclusion dictionary
            conclusion = {
                "concatenated_lexical_state": concatenated_sorted,
                "completion": intermediate_candidate[1] if len(intermediate_candidate) > 1 else None,
                "final_ranking": intermediate_candidate[0] if len(intermediate_candidate) > 0 else None,
                "final_prediction": final_prediction
            }

            # Log the current state, completion, and predicted answer
            logger.debug(
                f"Whole video state (i={i}): {answerable_concatenated_summaries}"
            )
            logger.debug(
                f"Completion (i={i}): {intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            )
            logger.debug(
                f"Predicted answer (i={i}): {intermediate_candidate[0] if len(intermediate_candidate) > 0 else 'N/A'}"
            )

            # Add the conclusion to the conclusions dictionary
            conclusions[i] = {
                "node_interval_center": interval_center,
                "conclusion": conclusion
            }

            # Create the reasoning history
            reasoning_history = (
                f"### Exam Task\n\n{intermediate_candidate[2] if len(intermediate_candidate) > 2 else 'N/A'}"
                f"### Student Answer\n\n{intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            )

            # Escape curly braces for formatting
            reasoning_history_escaped = reasoning_history.replace("{", "{{").replace("}", "}}")

            # Format the self_reflect_prompt with the reasoning_history
            formatted_prompt = self.self_reflect_prompt.format(reasoning_history=reasoning_history_escaped)

            # Call the API to get the self-reflection completion
            self_reflection_completion = api.get_completion_from_text(
                text=formatted_prompt,
                completion_start=self.self_reflect_completion_start,
                max_new_tokens=self.self_reflect_max_new_tokens,
                temperature=self.self_reflect_temperature
            )
            logger.debug(
                f"Derived LLM completion of self-reflected confidence (i={i}): {self_reflection_completion}"
            )

            # Define the priority of certain keywords
            keywords_in_priority_order = {
                'confidence': 1,
                'level': 2,
                'conf': 3,
                'answerability': 4,
                'confidence_level': 5,
                'confidence level': 6
            }

            # Parse the answer JSON to extract confidence
            candidate = parse_answer_json(
                text=self_reflection_completion,
                keywords_in_priority_order=keywords_in_priority_order,
                candidate_fun=get_single_number_candidates_from_text
            )

            # Determine the confidence level
            confidence = 1 if not candidate else candidate
            confidence = max(1, min(3, confidence))
            logger.debug(f"Parsed self-reflected confidence from completion (i={i}): {confidence}")

            # If confidence is 3, stop the iteration
            if confidence == 3:
                logger.debug("Confidence reached 3. Stopping iteration.")
                break

        # Step 6: Generate the final conclusion dictionary
        if conclusions:
            last_conclusion = list(conclusions.values())[-1]["conclusion"]
            final_conclusion = {
                "number_of_iterations": len(conclusions),
                "iteration_conclusions": conclusions,
                "whole_video_state": answerable_concatenated_summaries,
                "completion": last_conclusion.get("completion"),
                "final_ranking": last_conclusion.get("final_ranking"),
                "final_prediction": last_conclusion.get("final_prediction")
            }
        else:
            final_conclusion = {
                "number_of_iterations": 0,
                "iteration_conclusions": {},
                "whole_video_state": {},
                "completion": None,
                "final_ranking": None,
                "final_prediction": None
            }

        return final_conclusion
    ####################################################################################################################################3
    # Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.
#############################################################################################################################################
class IterativeMergeConclusion(Operation):
    def __init__(
        self,
        qa_prompt: str,
        qa_completion_start: str,
        qa_max_new_tokens: int,
        qa_temperature: float,
        qa_replace_c: bool,
        qa_parse_strategy: str,
        self_reflect_prompt: str,
        self_reflect_completion_start: str,
        self_reflect_max_new_tokens: int,
        self_reflect_temperature: float,
        self_reflect_parse_strategy: str
    ):
        super().__init__()

        # Initialize QA member variables
        self.qa_prompt = qa_prompt
        self.qa_completion_start = qa_completion_start
        self.qa_max_new_tokens = qa_max_new_tokens
        self.qa_temperature = qa_temperature
        self.qa_replace_c = qa_replace_c
        self.qa_parse_strategy = qa_parse_strategy

        # Initialize self-reflection member variables
        self.self_reflect_prompt = self_reflect_prompt
        self.self_reflect_completion_start = self_reflect_completion_start
        self.self_reflect_max_new_tokens = self_reflect_max_new_tokens
        self.self_reflect_temperature = self_reflect_temperature
        self.self_reflect_parse_strategy = self_reflect_parse_strategy

    def _execute(
        self,
        graph: Optional['Graph'],
        api: Optional['API'],
        target: Optional['Node']
    ) -> Dict[str, Any]:
        # Step 1: Retrieve all answerable nodes with non-None ranking_confidence
        answerable_nodes: List['Node'] = [
            node for node in graph.get_concludable_nodes()
            if node.state.ranking_confidence is not None
        ]

        # Step 2: Extract whole video summary and task from the graph
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        # Step 3: Create a shallow copy and sort the answerable nodes by ranking_confidence (descending)
        sorted_answerable_nodes = list(answerable_nodes)
        sorted_answerable_nodes.sort(
            key=lambda x: x.state.ranking_confidence,
            reverse=True
        )

        # Step 4: Initialize summaries and conclusions
        answerable_concatenated_summaries: Dict[float, str] = {}
        conclusions: Dict[int, Dict[str, Any]] = {}

        # Step 5: Iterate over the sorted answerable nodes
        for i, node in enumerate(sorted_answerable_nodes):
            # Calculate the center point of the video clip interval
            interval_start = node.state.video_clip.sampled_indices[0]
            interval_end = node.state.video_clip.sampled_indices[-1]
            interval_center = (interval_start + interval_end) / 2

            # Add the node's lexical representation to the summaries
            lexical_representation = node.state.get_lexical_representation()
            answerable_concatenated_summaries[interval_center] = lexical_representation

            # Experimentally tested with llama 3 that there is almost 3% performance increase when collecting
            # all nodes with level 3 ranking_confidence before doing the first merge and inference
            if i != len(sorted_answerable_nodes) - 1:
                # Continue as long as the next node has a ranking_confidence of 3
                if sorted_answerable_nodes[i + 1].state.ranking_confidence == 3:
                    continue

            # Sort the answerable_concatenated_summaries by the center of the interval
            sorted_concatenated_summaries = dict(sorted(answerable_concatenated_summaries.items()))

            # Concatenate the lexical node states
            concatenated_lexical_state = "\n".join(sorted_concatenated_summaries.values())

            # Decide which candidate derivation function to use based on availability of the task options
            if all(option == "N/A" for option in task.options.values()):
                # Get the free-form QA prediction
                intermediate_candidate = derive_free_form_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_lexical_state,
                    question=task.question,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = intermediate_candidate[0]
            else:
                # Get the single-choice QA prediction
                intermediate_candidate = derive_options_candidate_from_whole_video_state(
                    whole_video_summary=whole_video_summary,
                    whole_video_state=concatenated_lexical_state,
                    task=task,
                    api=api,
                    prompt_template=self.qa_prompt,
                    completion_start=self.qa_completion_start,
                    max_new_tokens=self.qa_max_new_tokens,
                    replace_c=self.qa_replace_c,
                    parse_strategy=self.qa_parse_strategy,
                    temperature=self.qa_temperature
                )

                final_prediction = (
                    intermediate_candidate[0][0]
                    if intermediate_candidate[0]
                    else "failed to parse prediction from completion"
                )

            # Create the conclusion dictionary
            conclusion = {
                "concatenated_lexical_state": concatenated_lexical_state,
                "completion": intermediate_candidate[1] if len(intermediate_candidate) > 1 else None,
                "final_ranking": intermediate_candidate[0] if len(intermediate_candidate) > 0 else None,
                "final_prediction": final_prediction
            }

            # Log the current state, completion, and predicted answer
            logger.debug(
                f"Whole video state (i={i}): {answerable_concatenated_summaries}"
            )
            logger.debug(
                f"Completion (i={i}): {intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            )
            logger.debug(
                f"Predicted answer (i={i}): {intermediate_candidate[0] if len(intermediate_candidate) > 0 else 'N/A'}"
            )

            # Add the conclusion to the conclusions dictionary
            conclusions[i] = {
                "node_interval_center": interval_center,
                "conclusion": conclusion
            }

            # Combine the QA prompt with the completion to get the reasoning history
            reasoning_history = (
                f"### Exam Task\n\n{intermediate_candidate[2] if len(intermediate_candidate) > 2 else 'N/A'}"
                f"### Student Answer\n\n{intermediate_candidate[1] if len(intermediate_candidate) > 1 else 'N/A'}"
            )

            # Replace "{" with "{{" and "}" with "}}" to escape them in the prompt
            reasoning_history_escaped = reasoning_history.replace("{", "{{").replace("}", "}}")

            # Build the self-reflection prompt with the reasoning history
            formatted_self_reflect_prompt = self.self_reflect_prompt.format(
                reasoning_history=reasoning_history_escaped
            )

            # Get the completion of the self-reflection
            self_reflection_completion = api.get_completion_from_text(
                text=formatted_self_reflect_prompt,
                completion_start=self.self_reflect_completion_start,
                max_new_tokens=self.self_reflect_max_new_tokens,
                temperature=self.self_reflect_temperature
            )
            logger.debug(
                f"Derived LLM completion of self-reflected confidence (i={i}): {self_reflection_completion}"
            )

            # Define the priority of certain keywords
            keywords_in_priority_order = {
                'confidence': 1,
                'level': 2,
                'conf': 3,
                'answerability': 4,
                'confidence_level': 5,
                'confidence level': 6
            }

            # Parse the answerability from the completion
            candidate = parse_answer_json(
                text=self_reflection_completion,
                keywords_in_priority_order=keywords_in_priority_order,
                candidate_fun=get_single_number_candidates_from_text
            )

            # Assure valid numerical confidence
            confidence = 1 if not candidate else candidate
            confidence = max(1, min(3, confidence))
            logger.debug(f"Parsed self-reflected confidence from completion (i={i}): {confidence}")

            # If the confidence is 3, we can stop the iteration and use the last QA prediction
            if confidence == 3:
                logger.debug("Confidence reached 3. Stopping iteration.")
                break

        # Step 6: Generate the final conclusion dictionary
        if conclusions:
            last_conclusion = list(conclusions.values())[-1]["conclusion"]
            final_conclusion = {
                "number_of_iterations": len(conclusions),
                "iteration_conclusions": conclusions,
                "whole_video_state": answerable_concatenated_summaries,
                "completion": last_conclusion.get("completion"),
                "final_ranking": last_conclusion.get("final_ranking"),
                "final_prediction": last_conclusion.get("final_prediction")
            }
        else:
            final_conclusion = {
                "number_of_iterations": 0,
                "iteration_conclusions": {},
                "whole_video_state": {},
                "completion": None,
                "final_ranking": None,
                "final_prediction": None
            }

        logger.info(f"Executed LLM-based conclusion operation: IterativeMergeConclusion -> {final_conclusion['final_prediction']}")

        return final_conclusion