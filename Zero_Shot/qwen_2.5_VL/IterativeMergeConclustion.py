"""Please help me to write a class named IterativeMergeConclusion， it takes Operation as baseclass. and its inputs for the class are  qa_prompt: str, qa_completion_start: str,qa_max_new_tokens: int, qa_temperature: float, qa_replace_c: bool,qa_parse_strategy: str,self_reflect_prompt: str,self_reflect_completion_start: str,self_reflect_max_new_tokens: int,self_reflect_temperature: float,self_reflect_parse_strategy: str
and of course there are some member variables to be initialized: self.qa_prompt = qa_prompt, self.qa_completion_start = qa_completion_start,self.qa_max_new_tokens = qa_max_new_tokens, self.qa_temperature = qa_temperature,self.qa_replace_c = qa_replace_c,self.qa_parse_strategy = qa_parse_strategy,self.self_reflect_prompt = self_reflect_prompt, self.self_reflect_completion_start = self_reflect_completion_start,self.self_reflect_max_new_tokens = self_reflect_max_new_tokens,self.self_reflect_temperature = self_reflect_temperature, self.self_reflect_parse_strategy = self_reflect_parse_strategy
the function is named _execute, which has inputs including graph: Optional[Graph], api: Optional[API], target=Optional[Node] and it returns dict[str, any]
firstly get all answerable_nodes via graph.get_concludable_nodes(), and if the node.state.ranking_confidence is not none, you can add this node to the answerable_nodes list
then use graph.root.state.spatial_node_state.action_captions_summary to return whole_video_summary and use graph.root.state.task to return variable task.
then use list[] to get shallow copy of answerable_nodes. and next .sort() to sort answerable_nodes depending the nodes.state.ranking_confidence in list from highest to lowest.
initialize the answerable_concatenated_summaries with {} and following see the center point of interval is the most answerable node
initialize the conclusions {}
iterate over answerable_nodes using enumerate and get the center point of the interval of videoclip presented by node
interval_start from node.state.video_clip.sampled_indices[0]; end from node.state.video_clip.sampled_indices[-1]. And after adding them divide it by 2
add the node to the answerable_concatenated_summaries,corresponding key is interval_center ,value is node.state.get_lexical_representation()
next using if i current index for enumerate is not the last index and next element(node) in answerable_nodes 's state.ranking_confidence is 3 then continue this loop
then next dict（sorted) the answerable_concatenated_summaries and join "\n" in answerable_concatenated_summaries.values()
if all option in the task.options.values() are all "N/A" then all the function and returns the result as intermediate_candidate and set intermediate_candidate[0] to final_prediction
otherwise call derive_options_candidate_from_whole_video_state to get the result as intermediate_candidate and then if intermediate_candidate[0] exists and set final_prediction as intermediate_candidate[0][0], otherwise set it to "failed to parse prediction from completion"
create a dictionary named conclusion including "concatenated_lexical_state" "concatenated_lexical_state" ,"completion": intermediate_candidate[1],"final_ranking": intermediate_candidate[0],"final_prediction": final_prediction
and then do logger.debug "Whole video state (i={i}): {answerable_concatenated_summaries}" ,"Completion (i={i}): {intermediate_candidate[1]}","Predicted answer (i={i}): {intermediate_candidate[0]}"
and then set the conclusions[i] as "node_interval_center": interval_center,"conclusion": conclusion.
create variable reaning_history = f"### Exam Task\n\n{intermediate_candidate[2]}### Student Answer\n\n{intermediate_candidate[1]}"
and then use .replace() to replace "{" with "{{" and replace "}" with "}}".
next do format operation to self.self_reflect_prompt to replace the reasoning_history in prompt with just gotten reasoning_history
next call api.get_completion_from_text with the one of inputs(self_reflection_prompt) to get complection of self-reflection
define the priority of certain keywords as dictionary keywords_in_priority_order likes 'confidence','level','conf','answerability','confidence_level','confidence level'
then call parse_answer_json() using text=completion,keywords_in_priority_order=keywords_in_priority_order,candidate_fun=get_single_number_candidates_from_text as inputs to return candidate
can then this candidate is not true or not existing then set confidence as 1 othterwise set confidence as candidate
then make sure the confidence in range of 1-3 and if here confidence is equel to 3 then break stoping the iteration.
finally generate a dictionary named conclusion including "number_of_iterations": len(conclusions),"iteration_conclusions": conclusions,"whole_video_state": answerable_concatenated_summaries,"completion": list(conclusions.values())[-1]["conclusion"]["completion"],"final_ranking": list(conclusions.values())[-1]["conclusion"]["final_ranking"],"final_prediction": list(conclusions.values())[-1]["conclusion"]["final_prediction"]
this whole function finally returns the conclusion dictionary"""