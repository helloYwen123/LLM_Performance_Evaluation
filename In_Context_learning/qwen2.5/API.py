#######################################################
###################### instruction Prompt #############
#######################################################
""" The class is named API. It has several inputs  get_object_detections_from_video_clip_and_text_config: dict[str, any],get_action_captions_from_video_clip_config: dict[str, any],get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],get_summary_from_noisy_perceptive_data_config: dict[str, any],get_unspecific_objects_from_video_clip_config: dict[str, any],get_completion_from_text_config: dict[str, any],get_specific_objects_from_video_clip_config: dict[str, any],random_seed: int,reset_seed_for_each_function: bool = True,load_models_only_when_needed: bool = False
and then initialize its member variables with these inputs, member variables includes: self.get_object_detections_from_video_clip_and_text_config;self.get_action_captions_from_video_clip_config;self.get_temporal_grounding_from_video_clip_and_text_config;self.get_summary_from_noisy_perceptive_data_config;self.get_completion_from_text_config;self.get_unspecific_objects_from_video_clip_config;self.get_specific_objects_from_video_clip_config;self.load_models_only_when_needed;self.random_seed;self.reset_seed_for_each_function
then run _initialize_llm() in __init__
\nThe first function is named __initialize__llm(), no inputs and outputs. At first, it creates a dictionary named available_llm_classes, which has keys "HuggingFaceLLM" and "OpenAILLM" and respective value are two initializing functions: initialize_huggingface_llm_from_config and initialize_openai_llm_from_config, which are two functions from two classes(HuggingFaceLLM and OpenAILLM).
then function assigns variable self.llm with this dictionary's value depending on get_completion_from_text_config. The key is from the value under the get_completion_from_text_config's key[llm_class]. The value is directly get_completion_from_text_config. The If self.load_models_only_when_needed is false then self.llm.build_model(). 
\nThe second function is named reset_seed; need to first clear the cache for cuda or anything else.
if not self.reset_seed_for_each_function then info Random seed is not reset for each function call.
then set ramdon seed for numpy, cuda.
\nThe third function is named get_object_detections_from_video_clip_and_text. It takes video_clip: VideoClip as input
first run self.reset_seed in this function and finally returns a instance named infer_bounding_boxes_from_video
\nThe fourth function is named get_action_captions_from_video_clip 
run self.reset_seed(). Then initialize final_action_captions as None and if self.get_action_captions_from_video_clip_config["model_name"] is not one of "LaViLa" or "CogAgent" then report error
if self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"] is not None , then # load pre-inferred action captions if a path for them is given in the config via two functions : read_json_file and get_clip_data_from_video_data
if self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa" then get sample_rate from get_action_captions_from_video_clip_config["resample_rate"]
sequentially get video_clip_data and start_frame and then depending on the value self.get_action_captions_from_video_clip_config["model_name"] is "LaViLa" or "CogAgent" to call their corresponding fuction named infer_transcript_from_video_clip_using_action_captions and infer_transcript_from_video_clip_using_frame_captions from lavila_video_captioner.narrator_inference and CogAgent.video's module
this function needs inputs from get_action_captions_from_video_clip_config
the result is saved as variable named action_captions and then reduce to the very first action caption of each interval for now and then save this regenerated results as final_action_captions
then use assert to make sure final_action_captions is not none. The function finally returns list of inal_action_captions.values()
\nThe fifth function is get_temporal_grounding_from_video_clip_and_text, it takes video_clip: VideoClip and text: str as inputs
firstly run also reset_seed() and then call infer_temporal_grounding_score_from_video_and_text to get result as text_temporal_grounding
text_temporal_grounding is a dictionary with keys foreground_indicators, boundary_offsets,saliency_scores and then respectively save the value in these keys
make a list of the foreground indicators. then derive the best k boundaries offset indices, here k is from get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
and then flatten foreground_indicators to pick top-k indices saving as top_k_indices 
next initialize relevance_indicators with 0 in length as video_clip
iteratively pick top k interval. Here top_k_indices is just the boundary_offsets' index list. with the help of top_k_indices it can get top_i_boundary_offset
depending on top_i_boundary_offset[0] and top_i_boundary_offset[1] gets it the optimistic flooring of start index and  optimistic ceiling of end index respectively as start_index and end_index.
update the relevance indicators and set all relevance indicators between start and end index to 1
Then use logger.debug(f"Derived relevance indicators: {relevance_indicators}").
Finally, initialize the salience_indicators as empty list. If if saliency_scores.dim() == 0 then num_saliency_scores = 0 otherwise set it to saliency_scores.size(0)
Then do iteration in range of num_saliency_scores. In i-iteration, assigning the saliency_scores i-th element to saliency_score. At the same time,  use max and min function to limit score between 0 and 1. Each saliency_score will be appended into salience_indicators. This is followed by the code logger.debug("Derived salience indicators").
this whole function will return the dictionary with keys: "foreground_indicators","relevance_indicators","salience_indicators".  """