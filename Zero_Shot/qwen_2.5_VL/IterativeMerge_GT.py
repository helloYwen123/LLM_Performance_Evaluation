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