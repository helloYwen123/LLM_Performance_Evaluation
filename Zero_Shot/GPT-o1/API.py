#Ground Truth


# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# please help me to finish a class named API. It has several inputs  get_object_detections_from_video_clip_and_text_config: dict[str, any],get_action_captions_from_video_clip_config: dict[str, any],get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],get_summary_from_noisy_perceptive_data_config: dict[str, any],get_unspecific_objects_from_video_clip_config: dict[str, any],get_completion_from_text_config: dict[str, any],get_specific_objects_from_video_clip_config: dict[str, any],random_seed: int,reset_seed_for_each_function: bool = True,load_models_only_when_needed: bool = False
# and then initialize its member variables with these inputs, member variables includes: self.get_object_detections_from_video_clip_and_text_config;self.get_action_captions_from_video_clip_config;self.get_temporal_grounding_from_video_clip_and_text_config;self.get_summary_from_noisy_perceptive_data_config;self.get_completion_from_text_config;self.get_unspecific_objects_from_video_clip_config;self.get_specific_objects_from_video_clip_config;self.load_models_only_when_needed;self.random_seed;self.reset_seed_for_each_function
# then run _initialize_llm() in __init__
# the first function is named __initialize__llm(), no inputs and outputs
# creating a dictionary named available_llm_classes, which has keys "HuggingFaceLLM" and "OpenAILLM" and respective value are  HuggingFaceLLM.initialize_huggingface_llm_from_config and OpenAILLM.initialize_openai_llm_from_config, which are two functions in two class
# assign self.llm with this dictionary's value depending on get_completion_from_text_config
# if self.load_models_only_when_needed is false then self.llm.build_model()
# the second function is named reset_seed; need to first clear the cache for cuda or anything else.
# if not self.reset_seed_for_each_function then info Random seed is not reset for each function call.
# then set ramdon seed for numpy, cuda
# the third function is named get_object_detections_from_video_clip_and_text. It takes video_clip: VideoClip as input
# first run self.reset_seed in this function and finally returns a instance named infer_bounding_boxes_from_video 
# the fourth function is named get_action_captions_from_video_clip 
# run self.reset_seed()
# initialize final_action_captions as None and if self.get_action_captions_from_video_clip_config["model_name"] is not one of "LaViLa" or "CogAgent" then report error
# if self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"] is not None , then # load pre-inferred action captions if a path for them is given in the config via two functions : read_json_file and get_clip_data_from_video_data
# if self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa" then get sample_rate from get_action_captions_from_video_clip_config["resample_rate"]
# sequentially get video_clip_data and start_frame 
# and then depending on the value self.get_action_captions_from_video_clip_config["model_name"] is "LaViLa" or "CogAgent" to call their corresponding fuction named infer_transcript_from_video_clip_using_action_captions and infer_transcript_from_video_clip_using_frame_captions from lavila_video_captioner.narrator_inference and CogAgent.video's module
# this function needs inputs from get_action_captions_from_video_clip_config
# the result is saved as variable named action_captions and then reduce to the very first action caption of each interval for now and then save this regenerated results as final_action_captions
# then use assert to make sure final_action_captions is not none. The function finally returns list of inal_action_captions.values()
# the fourth function is get_temporal_grounding_from_video_clip_and_text, it takes video_clip: VideoClip and text: str as inputs
# firstly run also reset_seed() and then call infer_temporal_grounding_score_from_video_and_text to get result as text_temporal_grounding
# text_temporal_grounding is a dictionary with keys foreground_indicators, boundary_offsets,saliency_scores and then respectively save the value in these keys
# make a list of the foreground indicators. 
# then derive the best k boundaries offset indices, here k is from get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
# and then flatten foreground_indicators to pick top-k indices saving as top_k_indices 
# next initialize relevance_indicators with 0 in length as video_clip
# iteratively pick top k interval. Here top_k_indices is just the boundary_offsets' index list. with the help of top_k_indices it can get top_i_boundary_offset
# depending on top_i_boundary_offset[0] and top_i_boundary_offset[1] gets it the optimistic flooring of start index and  optimistic ceiling of end index respectively as start_index and end_index.
# # update the relevance indicators and set all relevance indicators between start and end index to 1
# finally logger.debug(f"Derived relevance indicators: {relevance_indicators}")
# initialize the salience_indicators as empty list. If if saliency_scores.dim() == 0 then num_saliency_scores = 0 otherwise set it to saliency_scores.size(0)
# If num_saliency_scores is larger than 1, the saliency_scores will be retrieve the i-th score. if the number of saliency_score is 1 then direct to assgin it to saliency_score.
# and the saliency_score's element must be between 0 and 1
# the final function will return the dictionary with keys: "foreground_indicators","relevance_indicators","salience_indicators"
# The fifth function is named get_summary_from_noisy_perceptive_data, it takes action_captions: list[str],object_detections: list[str],temporal_grounding: list[str],interleaved_action_captions_and_object_detections: list[str],video_clip: VideoClip,question: Optional[str] = "", options: Optional[dict[str, str]] = None,words: int = 500 as its inputs
# run self.reset_seed() firstly,
# if self.get_summary_from_noisy_perceptive_data_config["pre_inferred_summaries_path"] is not None,# load pre-inferred summaries json file via function read_json_file
# get the summaries entry for the video clip from the generated above result summaries
# get the frame interval of the clip using video_clip.sampled_indices
# define a tolerance threshold for the frame interval (due to noise in sampling process and rounding), here set it as video_clip.original_fps // 2
# if start_frame == whole_video_start_frame and end_frame == whole_video_end_frame then go into the multi-if-branch
# if len(action_captions) != 0 then return video_summaries["root_action_captions_summaries"]
# if len(object_detections) != 0 then return video_summaries["root_object_detections_summaries"]
# if len(temporal_grounding) != 0 then return video_summaries["root_temporal_grounding_summaries"]
# if other situations occur then raise error
# initialize index = None and enumerate(video_summaries["clip_boundaries"] to get i and clip_boundary iteratively
# check if the clip boundary is within the frame interval taking account into ttolerance then set index = i
# then if index after process is none , then raise error
# find out what type of summary depending on the length of action_captions,object_detections and temporal_grounding
# if len(action_captions) != 0 then summary_type = "action_caption_summaries" and fallback_summary_type = "n.a."
# if len(object_detections) != 0 then summary_type = "object_detections_summaries" and fallback_summary_type = "unspecific_object_detection_summaries"
# if len(temporal_grounding) != 0 then summary_type = "temporal_grounding_summaries" and fallback_summary_type = "n.a."
# raise error finally if no satisfied
# get the defined summary for the clip and if if the defined summary is not available, use the fallback summary 
# and set summary to summaried[index] then returns summary
# if len(action_captions) != 0 , then get prompt_template and completion_start from dictionary self.get_summary_from_noisy_perceptive_data_config
# do the same process to  object_detections, temporal_grounding and interleaved_action_captions_and_object_detections, whose variables can be easily derived.
# In these four process the other three of the length of object_detections, temporal_grounding, interleaved_action_captions_and_object_detections and temporal_grounding are supposed to be 0 for one of situation. Here assert is used
# if other situations occur then raising error
# if options is None then setting options as a dictionary with 4 elements for the "options 0,1" etc.
# if question is not None and self.get_summary_from_noisy_perceptive_data_config.get("replace_c", True) then run question = replace_c_with_camera_wearer(question)
# next the function named get_summary_from_noisy_perceptive_data has also a subfunction named chunks, which takes chunk_size as inputs , it will generate several slices for chunk_size in length of data. Finally, using yield to return data[j:j + chunk_size]
# do while for length of noisy_data is not 1, initialize the new_noisy_data as [] and caption_i as 0;
# call chunks in for-loop here chunks takes noisy_data and chunk_size=self.get_summary_from_noisy_perceptive_data_config.get("interval_span") as inputs.
# remove dots at end of caption ends if they exist and if if not len(temporal_grounding) != 0 sets the delimiter as ". " otherwise as ""
# then use join() to join noisy_data_chunk with delimiter
# then create and fill prompt with  prompt_template.format, which inputs include interval_text, question, option_0-4,length and words
# then call self.get_completion_from_text to create the variable summary
# then do summary = summary.strip()
# if self.get_summary_from_noisy_perceptive_data_config["no_recursion"] then Remove Ending Period for summary and Add Interval Format.
# if self.get_summary_from_noisy_perceptive_data_config.get("remove_linebreaks") then replace linebreaks with whitespaces and append the precessed summary to new_noisy_data 
# after iterate all noise data , reset the noisy_data with new_noisy_data
# it will do break in the whole while loop if self.get_summary_from_noisy_perceptive_data_config["no_recursion"]
# and after while if not self.get_summary_from_noisy_perceptive_data_config["no_recursion"] then use assert to make sure length of noisy_data as 1. and then set summary as noisy_data[0]. Otherwise using join " " to noisy_data
# the whole function named get_summary_from_noisy_perceptive_data will return the processed variable summary
# Next function of class is named get_completion_from_text，which takes self, text: str, completion_start: str, max_new_tokens: Optional[int] and temperature: Optional[float] as its inputs
# run slef.reset_seed firstly and if self.load_models_only_when_needed is true then run self.llm.build_model()
# then call self.llm.get_completion and set result as completion, here self.llm is one instance for class
# after this run self.llm.destroy_model() to free up GPU space and avoid OOM errors on limited GPU. finally function return completion
# the next one function is named get_unspecific_objects_from_video_clip, which takes video_clip: VideoClip as input
# firstly, run reset_seed
# if self.get_unspecific_objects_from_video_clip_config["pre_inferred_object_detections_path"] then read_json_file and pick object_detections from object_detections[video_clip.id];
# run get_clip_data_from_video_data to get final_object_detecctions
# if 'get_clip_data_from_video_data' is false then  get unspecific objects for the video_clip_data using CogAgent library's function
# the result is objects_per_frame ,reduce to the very first action caption of each interval for now ,which means only take first entry of every element in objects_per_frame.value(), take the result as raw_object_detections
# then run  raw_object_detections = dict(zip(objects_per_frame.keys(), raw_object_detections))
# initialize the final_object_detection as {} and interatively to pick element in raw_object_detections.items() as interval and competion
# assuming that we prompted the model to list objects using parse_list(completion)
# only use the specified number of objects per frame and using assert to make sure self.get_unspecific_objects_from_video_clip_config['num_objects_per_frame'] > 0 
# get the specified number of objects based on config in get_unspecific_objects_from_video_clip_config
# remove dots and linebreaks from objects and strip each object and remove commas from the ends of each object and remove ", and" from the ends of each object
# if length of object_in_completion larger than 0 and set objects as []
# save it in dictionary as final_object_detections[interval] = objects
# return the list of dictionary's value
# the last function is get_specific_object_confidences_from_video_clip, which takes inputs:video_clip: VideoClip, question: str,options: dict[str, str]
# run reset function self.reset_seed()
# if self.get_specific_objects_from_video_clip_config.get("replace_c", True) then run  question = replace_c_with_camera_wearer(question) and options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}
# and format the prompt and get completion variable prediciton from llm via self.get_completion_from_text, whose inputs includes text=prompt,completion_start=completion_start,max_new_tokens=self.get_specific_objects_from_video_clip_config["max_new_tokens"],temperature=self.get_specific_objects_from_video_clip_config["temperature"]
# parse the completion using parse_list
# filter objects via filter_list_of_objects
# initialize confidences_per_object with {} and then iterate the element in objects to call self.get_object_detections_from_video_clip_and_text(video_clip=video_clip, text=o) and set result as object_detections
# calculate mean confidence of the object detection and then run mean_confidence = 1 / (1 + np.exp(-10 * (mean_confidence - 0.3)))
# saving as confidences_per_object[object] = mean_confidence
# return finally confidences_per_object

###############################################################################
class API:
    def __init__(
        self,
        get_object_detections_from_video_clip_and_text_config: Dict[str, Any],
        get_action_captions_from_video_clip_config: Dict[str, Any],
        get_temporal_grounding_from_video_clip_and_text_config: Dict[str, Any],
        get_summary_from_noisy_perceptive_data_config: Dict[str, Any],
        get_unspecific_objects_from_video_clip_config: Dict[str, Any],
        get_completion_from_text_config: Dict[str, Any],
        get_specific_objects_from_video_clip_config: Dict[str, Any],
        random_seed: int,
        reset_seed_for_each_function: bool = True,
        load_models_only_when_needed: bool = False
    ):
        # Initialize member variables
        self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
        self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
        self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
        self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config
        self.get_completion_from_text_config = get_completion_from_text_config
        self.get_unspecific_objects_from_video_clip_config = get_unspecific_objects_from_video_clip_config
        self.get_specific_objects_from_video_clip_config = get_specific_objects_from_video_clip_config
        self.random_seed = random_seed
        self.reset_seed_for_each_function = reset_seed_for_each_function
        self.load_models_only_when_needed = load_models_only_when_needed

        # Initialize LLM
        self._initialize_llm()

    def _initialize_llm(self):
        available_llm_classes = {
            "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        llm_type = self.get_completion_from_text_config.get("llm_type")
        llm_config = self.get_completion_from_text_config.get("config", {})

        if llm_type not in available_llm_classes:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        # Initialize the appropriate LLM
        self.llm = available_llm_classes[llm_type](llm_config)

        if not self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built during initialization.")

    def reset_seed(self):
        # Clear CUDA cache and other caches if necessary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        # Add any other cache clearing mechanisms here

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        # Set random seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        logger.debug(f"Random seed set to {self.random_seed}.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip) -> Any:
        self.reset_seed()
        # Placeholder for actual object detection inference
        # Replace with actual implementation
        infer_bounding_boxes_from_video = {"detections": []}  # Example structure
        return infer_bounding_boxes_from_video

    def get_action_captions_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        self.reset_seed()
        final_action_captions = None

        model_name = self.get_action_captions_from_video_clip_config.get("model_name")
        if model_name not in ["LaViLa", "CogAgent"]:
            raise ValueError(f"Unsupported model name for action captions: {model_name}")

        pre_inferred_path = self.get_action_captions_from_video_clip_config.get("pre_inferred_action_captions_path")
        if pre_inferred_path is not None:
            action_captions_data = read_json_file(pre_inferred_path)
            final_action_captions = self.get_clip_data_from_video_data(action_captions_data)
            logger.debug("Loaded pre-inferred action captions from path.")

        if final_action_captions is None:
            sample_rate = self.get_action_captions_from_video_clip_config.get("resample_rate", 1)
            video_clip_data = get_clip_data_from_video_data({"clip_data": video_clip.sampled_indices})
            start_frame = video_clip.sampled_indices[0]

            if model_name == "LaViLa":
                from lavila_video_captioner.narrator_inference import infer_transcript_from_video_clip_using_action_captions
                action_captions = infer_transcript_from_video_clip_using_action_captions(
                    video_clip=video_clip,
                    video_clip_data=video_clip_data,
                    start_frame=start_frame,
                    sample_rate=sample_rate,
                    config=self.get_action_captions_from_video_clip_config
                )
            elif model_name == "CogAgent":
                from CogAgent.video import infer_transcript_from_video_clip_using_frame_captions
                action_captions = infer_transcript_from_video_clip_using_frame_captions(
                    video_clip=video_clip,
                    video_clip_data=video_clip_data,
                    start_frame=start_frame,
                    sample_rate=sample_rate,
                    config=self.get_action_captions_from_video_clip_config
                )
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            # Reduce to the very first action caption of each interval
            final_action_captions = {k: v[0] for k, v in action_captions.items()}
            assert final_action_captions is not None, "Final action captions should not be None."
            logger.debug("Action captions generated and reduced to first of each interval.")

        return list(final_action_captions.values())

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Dict[str, Any]:
        self.reset_seed()
        # Placeholder for actual temporal grounding inference
        text_temporal_grounding = self.infer_temporal_grounding_score_from_video_and_text(video_clip, text)

        foreground_indicators = text_temporal_grounding.get("foreground_indicators", [])
        boundary_offsets = text_temporal_grounding.get("boundary_offsets", [])
        saliency_scores = text_temporal_grounding.get("saliency_scores", [])

        # Derive top-k boundaries
        top_k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        top_k_indices = sorted(range(len(foreground_indicators)), key=lambda i: foreground_indicators[i], reverse=True)[:top_k]

        relevance_indicators = [0] * len(video_clip.sampled_indices)
        for idx in top_k_indices:
            start_offset, end_offset = boundary_offsets[idx]
            start_index = max(0, start_offset)  # Optimistic flooring
            end_index = min(len(video_clip.sampled_indices), end_offset)  # Optimistic ceiling
            for i in range(start_index, end_index):
                relevance_indicators[i] = 1

        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        # Handle saliency scores
        salience_indicators = []
        if hasattr(saliency_scores, 'dim') and saliency_scores.dim() == 0:
            num_saliency_scores = 0
        else:
            num_saliency_scores = saliency_scores.size(0) if hasattr(saliency_scores, 'size') else len(saliency_scores)

        if num_saliency_scores > 1:
            salience_indicators = [score.item() for score in saliency_scores]
        elif num_saliency_scores == 1:
            salience_indicators = [saliency_scores.item()]
        else:
            salience_indicators = []

        # Ensure saliency scores are between 0 and 1
        salience_indicators = [max(0.0, min(1.0, score)) for score in salience_indicators]

        return {
            "foreground_indicators": foreground_indicators,
            "relevance_indicators": relevance_indicators,
            "salience_indicators": salience_indicators
        }

    def infer_temporal_grounding_score_from_video_and_text(self, video_clip: VideoClip, text: str) -> Dict[str, Any]:
        # Placeholder for the actual implementation
        return {
            "foreground_indicators": np.random.randint(0, 2, size=10).tolist(),
            "boundary_offsets": [(i, i+1) for i in range(10)],
            "saliency_scores": torch.tensor([0.5] * 10)
        }

    def get_summary_from_noisy_perceptive_data(
        self,
        action_captions: List[str],
        object_detections: List[str],
        temporal_grounding: List[str],
        interleaved_action_captions_and_object_detections: List[str],
        video_clip: VideoClip,
        question: Optional[str] = "",
        options: Optional[Dict[str, str]] = None,
        words: int = 500
    ) -> str:
        self.reset_seed()

        pre_inferred_summaries_path = self.get_summary_from_noisy_perceptive_data_config.get("pre_inferred_summaries_path")
        if pre_inferred_summaries_path is not None:
            summaries_data = read_json_file(pre_inferred_summaries_path)
            video_summaries = summaries_data.get(video_clip.id, {})
            logger.debug("Loaded pre-inferred summaries from path.")

            # Get frame interval
            sampled_indices = video_clip.sampled_indices
            start_frame = sampled_indices[0]
            end_frame = sampled_indices[-1]
            tolerance = video_clip.original_fps // 2

            whole_video_start_frame = 0  # Define appropriately
            whole_video_end_frame = int(video_clip.duration * video_clip.original_fps)

            if start_frame == whole_video_start_frame and end_frame == whole_video_end_frame:
                if len(action_captions) != 0:
                    return video_summaries.get("root_action_captions_summaries", "")
                if len(object_detections) != 0:
                    return video_summaries.get("root_object_detections_summaries", "")
                if len(temporal_grounding) != 0:
                    return video_summaries.get("root_temporal_grounding_summaries", "")
                raise ValueError("No valid summaries found for the provided data.")

            index = None
            for i, clip_boundary in enumerate(video_summaries.get("clip_boundaries", [])):
                if abs(clip_boundary[0] - start_frame) <= tolerance and abs(clip_boundary[1] - end_frame) <= tolerance:
                    index = i
                    break

            if index is None:
                raise ValueError("No matching clip boundary found within tolerance.")

            if len(action_captions) != 0:
                summary_type = "action_caption_summaries"
                fallback_summary_type = "n.a."
            elif len(object_detections) != 0:
                summary_type = "object_detections_summaries"
                fallback_summary_type = "unspecific_object_detection_summaries"
            elif len(temporal_grounding) != 0:
                summary_type = "temporal_grounding_summaries"
                fallback_summary_type = "n.a."
            else:
                raise ValueError("No valid data provided for summary.")

            summary = video_summaries.get(summary_type, video_summaries.get(fallback_summary_type, ""))
            return summary

        # If pre-inferred summaries are not available, generate them
        prompt_template = self.get_summary_from_noisy_perceptive_data_config.get("prompt_template", "{}")
        completion_start = self.get_summary_from_noisy_perceptive_data_config.get("completion_start", "")
        interval_span = self.get_summary_from_noisy_perceptive_data_config.get("interval_span", 10)
        no_recursion = self.get_summary_from_noisy_perceptive_data_config.get("no_recursion", False)
        remove_linebreaks = self.get_summary_from_noisy_perceptive_data_config.get("remove_linebreaks", False)

        if options is None:
            options = {f"option_{i}": f"Option {i}" for i in range(4)}

        if question and self.get_summary_from_noisy_perceptive_data_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)

        def chunks(data: List[str], chunk_size: int):
            for j in range(0, len(data), chunk_size):
                yield data[j:j + chunk_size]

        noisy_data = action_captions + object_detections + temporal_grounding + interleaved_action_captions_and_object_detections

        while len(noisy_data) > 1:
            new_noisy_data = []
            for i, noisy_data_chunk in enumerate(chunks(noisy_data, chunk_size=interval_span)):
                # Remove trailing dots if necessary
                if not temporal_grounding:
                    delimiter = ". "
                    noisy_data_chunk = [caption.rstrip('.').strip() for caption in noisy_data_chunk]
                else:
                    delimiter = ""
                interval_text = delimiter.join(noisy_data_chunk)

                # Create prompt
                prompt = prompt_template.format(
                    interval_text=interval_text,
                    question=question,
                    **options,
                    length=len(noisy_data_chunk),
                    words=words
                )

                # Get completion
                summary = self.get_completion_from_text(
                    text=prompt,
                    completion_start=completion_start,
                    max_new_tokens=self.get_summary_from_noisy_perceptive_data_config.get("max_new_tokens"),
                    temperature=self.get_summary_from_noisy_perceptive_data_config.get("temperature")
                ).strip()

                if no_recursion:
                    summary = summary.rstrip('.')
                    summary += " Interval Format."

                if remove_linebreaks:
                    summary = summary.replace('\n', ' ')
                new_noisy_data.append(summary)

            noisy_data = new_noisy_data
            if no_recursion:
                break

        if not no_recursion:
            assert len(noisy_data) == 1, "Noisy data should be reduced to a single summary."
            summary = noisy_data[0]
        else:
            summary = " ".join(noisy_data)

        return summary

    def get_completion_from_text(
        self,
        text: str,
        completion_start: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        self.reset_seed()

        if self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built on demand.")

        completion = self.llm.get_completion(
            text=text,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        if self.load_models_only_when_needed:
            self.llm.destroy_model()
            logger.debug("LLM model destroyed after completion.")

        return completion

    def get_unspecific_objects_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        self.reset_seed()
        pre_inferred_path = self.get_unspecific_objects_from_video_clip_config.get("pre_inferred_object_detections_path")
        if pre_inferred_path:
            object_detections_data = read_json_file(pre_inferred_path)
            object_detections = object_detections_data.get(video_clip.id, [])
            final_object_detections = get_clip_data_from_video_data(object_detections)
            logger.debug("Loaded pre-inferred object detections from path.")
        else:
            # Placeholder for actual object detection using CogAgent
            objects_per_frame = {"frame1": ["object1", "object2"], "frame2": ["object3"]}
            raw_object_detections = {k: v[0] for k, v in objects_per_frame.items()}
            final_object_detections = list(raw_object_detections.values())
            logger.debug("Generated unspecific object detections using CogAgent.")

        return final_object_detections

    def get_specific_object_confidences_from_video_clip(
        self,
        video_clip: VideoClip,
        question: str,
        options: Dict[str, str]
    ) -> Dict[str, float]:
        self.reset_seed()

        if self.get_specific_objects_from_video_clip_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)
            options = {k: replace_c_with_camera_wearer(v) for k, v in options.items()}

        prompt_template = self.get_specific_objects_from_video_clip_config.get("prompt_template", "{}")
        completion_start = self.get_specific_objects_from_video_clip_config.get("completion_start", "")
        max_new_tokens = self.get_specific_objects_from_video_clip_config.get("max_new_tokens", 50)
        temperature = self.get_specific_objects_from_video_clip_config.get("temperature", 0.7)

        # Create prompt
        prompt = prompt_template.format(
            question=question,
            **options
        )

        # Get completion
        prediction = self.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Parse and filter objects
        objects = filter_list_of_objects(parse_list(prediction))

        confidences_per_object = {}
        for obj in objects:
            object_detections = self.get_object_detections_from_video_clip_and_text(video_clip=video_clip, text=obj)
            # Placeholder for calculating mean confidence
            mean_confidence = np.random.rand()  # Replace with actual calculation
            mean_confidence = 1 / (1 + np.exp(-10 * (mean_confidence - 0.3)))
            confidences_per_object[obj] = mean_confidence

        return confidences_per_object
    ###########################################################################################################################################
    class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: Dict[str, Any],
            get_action_captions_from_video_clip_config: Dict[str, Any],
            get_temporal_grounding_from_video_clip_and_text_config: Dict[str, Any],
            get_summary_from_noisy_perceptive_data_config: Dict[str, Any],
            get_unspecific_objects_from_video_clip_config: Dict[str, Any],
            get_completion_from_text_config: Dict[str, Any],
            get_specific_objects_from_video_clip_config: Dict[str, Any],
            random_seed: int,
            reset_seed_for_each_function: bool = True,
            load_models_only_when_needed: bool = False
    ):
        self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
        self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
        self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
        self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config
        self.get_completion_from_text_config = get_completion_from_text_config
        self.get_unspecific_objects_from_video_clip_config = get_unspecific_objects_from_video_clip_config
        self.get_specific_objects_from_video_clip_config = get_specific_objects_from_video_clip_config
        self.load_models_only_when_needed = load_models_only_when_needed
        self.random_seed = random_seed
        self.reset_seed_for_each_function = reset_seed_for_each_function

        self._initialize_llm()

    def _initialize_llm(self):
        available_llm_classes = {
            "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        llm_class = self.get_completion_from_text_config.get("llm_class")
        if llm_class not in available_llm_classes:
            raise ValueError(f"Unsupported LLM class: {llm_class}")

        self.llm = available_llm_classes[llm_class](self.get_completion_from_text_config)

        # if the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built during initialization.")

    def reset_seed(self):
        # Clear CUDA cache and invalidate caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        importlib.invalidate_caches()
        logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Any:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        detections = infer_bounding_boxes_from_video(
            video_tensor=video_clip.data,
            obj=text,
            config_file=self.get_object_detections_from_video_clip_and_text_config["config_file"],
            checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
            box_threshold=self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
            text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
            cuda=self.get_object_detections_from_video_clip_and_text_config["cuda"]
        )

        logger.debug(f"Object detections for '{text}': {detections}")
        return detections

    def get_action_captions_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        final_action_captions = None
        model_name = self.get_action_captions_from_video_clip_config.get("model_name")

        if model_name not in ["LaViLa", "CogAgent"]:
            raise ValueError(f"Model name {model_name} is not supported for action captioning.")

        pre_inferred_path = self.get_action_captions_from_video_clip_config.get("pre_inferred_action_captions_path")
        if pre_inferred_path is not None:
            # Load pre-inferred action captions if a path is given in the config
            all_action_captions = read_json_file(file_path=pre_inferred_path)
            action_captions = all_action_captions.get(video_clip.id, {})
            final_action_captions = get_clip_data_from_video_data(
                video_data=action_captions,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )
            logger.warning("Using pre-inferred action captions. The pre-inferred action captions should represent intervals of 1 second for each second of the video.")
        else:
            if model_name == "LaViLa":
                # Infer action captions from the video clip using LaViLa
                sample_rate = self.get_action_captions_from_video_clip_config.get("resample_rate", 1)
                resampled_video_clip = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
                video_clip_data = resampled_video_clip.data
                start_frame = video_clip.sampled_indices[0]

                action_captions = infer_transcript_from_video_clip_using_action_captions(
                    video_clip=video_clip_data,
                    start_frame=start_frame,
                    fps=sample_rate,
                    original_fps=video_clip.original_fps,
                    interval_in_seconds=self.get_action_captions_from_video_clip_config.get("interval_in_seconds", 1),
                    temperature=self.get_action_captions_from_video_clip_and_text_config.get("temperature", 0.7),
                    top_p=self.get_action_captions_from_video_clip_and_text_config.get("top_p", 0.9),
                    max_text_length=self.get_action_captions_from_video_clip_and_text_config.get("max_new_tokens", 50),
                    num_return_sequences=self.get_action_captions_from_video_clip_and_text_config.get("num_return_sequences", 1),
                    early_stopping=self.get_action_captions_from_video_clip_and_text_config.get("early_stopping", True),
                    num_seg=self.get_action_captions_from_video_clip_and_text_config.get("num_seg", 4),
                    cuda=True,
                    modelzoo_dir_path=self.get_action_captions_from_video_clip_and_text_config.get("modelzoo_dir_path", ""),
                    checkpoint_download_url=self.get_action_captions_from_video_clip_and_text_config.get("checkpoint_download_url", ""),
                    checkpoint_file=self.get_action_captions_from_video_clip_and_text_config.get("checkpoint", "")
                )
            elif model_name == "CogAgent":
                # Infer action captions from the video clip using CogAgent
                sample_rate = self.get_action_captions_from_video_clip_config.get("resample_rate", 1)
                video_clip_data = video_clip.data
                start_frame = video_clip.sampled_indices[0]

                action_captions = infer_transcript_from_video_clip_using_frame_captions(
                    video_clip=video_clip_data,
                    start_frame=start_frame,
                    fps=sample_rate,
                    original_fps=video_clip.original_fps,
                    frame_prompt=self.get_action_captions_from_video_clip_config.get("frame_prompt", ""),
                    model_id=self.get_action_captions_from_video_clip_config.get("model_id", ""),
                    tokenizer_id=self.get_action_captions_from_video_clip_and_text_config.get("tokenizer_id", ""),
                    device=self.get_action_captions_from_video_clip_and_text_config.get("device", "cuda"),
                    precision=self.get_action_captions_from_video_clip_and_text_config.get("precision", "fp16"),
                    quantization=self.get_action_captions_from_video_clip_and_text_config.get("quantization", "int8"),
                    temperature=self.get_action_captions_from_video_clip_and_text_config.get("temperature", 0.7),
                    max_new_tokens=self.get_action_captions_from_video_clip_and_text_config.get("max_new_tokens", 50),
                    do_sample=self.get_action_captions_from_video_clip_and_text_config.get("do_sample", True)
                )

            # Reduce to the very first action caption of each interval for now
            final_action_captions = [captions[0] for captions in action_captions.values()]
            logger.debug("Action captions generated and reduced to the first of each interval.")

        # Ensure final_action_captions is not None
        assert final_action_captions is not None, "Final action captions should not be None."
        return final_action_captions

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Dict[str, Any]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
            video=video_clip.data,
            text=text,
            config_dir=self.get_temporal_grounding_from_video_clip_and_text_config.get("config_dir", ""),
            checkpoint_path=self.get_temporal_grounding_from_video_clip_and_text_config.get("checkpoint", ""),
            clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config.get("clip_model_version", ""),
            output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config.get("output_feat_size", 512),
            half_precision=self.get_temporal_grounding_from_video_clip_and_text_config.get("half_precision", True),
            jit=self.get_temporal_grounding_from_video_clip_and_text_config.get("jit", False),
            resize_size=self.get_temporal_grounding_from_video_clip_and_text_config.get("resize_size", 224),
            gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config.get("gpu_id", 0)
        )

        foreground_indicators = text_temporal_grounding.get("foreground_indicators", [])
        boundary_offsets = text_temporal_grounding.get("boundary_offsets", [])
        saliency_scores = text_temporal_grounding.get("saliency_scores", [])

        # Derive the best k boundaries offset indices
        top_k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        top_k_indices = sorted(range(len(foreground_indicators)), key=lambda i: foreground_indicators[i], reverse=True)[:top_k]

        # Initialize relevance_indicators with 0 for each frame in video_clip
        relevance_indicators = [0] * len(video_clip.sampled_indices)

        for idx in top_k_indices:
            boundary_offset = boundary_offsets[idx]
            start_offset, end_offset = boundary_offset

            # Optimistic flooring and ceiling
            start_index = max(0, int(start_offset * len(video_clip.sampled_indices)))
            end_index = min(int(end_offset * len(video_clip.sampled_indices)), len(video_clip.sampled_indices) - 1)

            # Update relevance indicators
            for i in range(start_index, end_index + 1):
                relevance_indicators[i] = 1

        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        # Handle saliency scores
        salience_indicators = []
        if isinstance(saliency_scores, torch.Tensor):
            num_saliency_scores = saliency_scores.size(0) if saliency_scores.dim() > 0 else 1
            for i in range(num_saliency_scores):
                saliency_score = saliency_scores[i].item() if num_saliency_scores > 1 else saliency_scores.item()
                saliency_score = max(0.0, min(1.0, saliency_score))
                salience_indicators.append(saliency_score)
        else:
            # If saliency_scores is not a tensor, assume it's a list
            salience_indicators = [max(0.0, min(1.0, score)) for score in saliency_scores]

        logger.debug("Derived salience indicators.")

        return {
            "foreground_indicators": foreground_indicators,
            "relevance_indicators": relevance_indicators,
            "salience_indicators": salience_indicators
        }

    def get_summary_from_noisy_perceptive_data(
            self,
            action_captions: List[str],
            object_detections: List[str],
            temporal_grounding: List[str],
            interleaved_action_captions_and_object_detections: List[str],
            video_clip: VideoClip,
            question: Optional[str] = "",
            options: Optional[Dict[str, str]] = None,
            words: int = 500
    ) -> str:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        pre_inferred_summaries_path = self.get_summary_from_noisy_perceptive_data_config.get("pre_inferred_summaries_path")
        if pre_inferred_summaries_path is not None:
            logger.debug("Using pre-inferred summaries.")

            # Load pre-inferred summaries JSON file
            summaries = read_json_file(file_path=pre_inferred_summaries_path)

            # Get the summaries entry for the video clip
            video_summaries = summaries.get(video_clip.id, {})
            logger.debug(f"Loaded summaries for video_clip ID {video_clip.id}.")

            # Get the frame interval of the clip
            sampled_indices = video_clip.sampled_indices
            start_frame = sampled_indices[0]
            end_frame = sampled_indices[-1]

            # Define a tolerance threshold for the frame interval
            tolerance = video_clip.original_fps // 2

            # Define the whole video start and end frame
            whole_video_start_frame = 0
            whole_video_end_frame = video_clip.original_num_frames - 1

            # If the video clip covers the whole video, try to take the whole video summary first
            if start_frame == whole_video_start_frame and end_frame == whole_video_end_frame:
                try:
                    if len(action_captions) != 0:
                        return video_summaries["root_action_captions_summaries"]
                    elif len(object_detections) != 0:
                        return video_summaries["root_object_detections_summaries"]
                    elif len(temporal_grounding) != 0:
                        return video_summaries["root_temporal_grounding_summaries"]
                    else:
                        raise ValueError("Please provide action captions, object detections, or temporal grounding to summarize.")
                except KeyError:
                    logger.warning("No whole video summary found. Trying to find the summary for the clip as long as the video.")

            # If not the whole video, find the matching clip summary
            index = None
            for i, clip_boundary in enumerate(video_summaries.get("clip_boundaries", [])):
                clip_start, clip_end = clip_boundary
                if abs(clip_start - start_frame) <= tolerance and abs(clip_end - end_frame) <= tolerance:
                    index = i
                    break

            if index is None:
                raise ValueError("No matching clip boundary found within tolerance.")

            if len(action_captions) != 0:
                summary_type = "action_caption_summaries"
                fallback_summary_type = "n.a."
            elif len(object_detections) != 0:
                summary_type = "object_detections_summaries"
                fallback_summary_type = "unspecific_object_detection_summaries"
            elif len(temporal_grounding) != 0:
                summary_type = "temporal_grounding_summaries"
                fallback_summary_type = "n.a."
            else:
                raise ValueError("No valid data provided for summary.")

            summary = video_summaries.get(summary_type, video_summaries.get(fallback_summary_type, ""))
            return summary

        # If pre-inferred summaries are not available, generate them
        prompt_template = self.get_summary_from_noisy_perceptive_data_config.get("prompt_template", "{}")
        completion_start = self.get_summary_from_noisy_perceptive_data_config.get("completion_start", "")
        interval_span = self.get_summary_from_noisy_perceptive_data_config.get("interval_span", 10)
        no_recursion = self.get_summary_from_noisy_perceptive_data_config.get("no_recursion", False)
        remove_linebreaks = self.get_summary_from_noisy_perceptive_data_config.get("remove_linebreaks", False)

        if options is None:
            options = {f"option_{i}": f"Option {i}" for i in range(4)}

        if question and self.get_summary_from_noisy_perceptive_data_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)

        def chunks(data: List[str], chunk_size: int):
            for j in range(0, len(data), chunk_size):
                yield data[j:j + chunk_size]

        noisy_data = action_captions + object_detections + temporal_grounding + interleaved_action_captions_and_object_detections

        while len(noisy_data) > 1:
            new_noisy_data = []
            for i, noisy_data_chunk in enumerate(chunks(noisy_data, chunk_size=interval_span)):
                # Remove trailing dots if necessary
                if not temporal_grounding:
                    delimiter = ". "
                    noisy_data_chunk = [caption.rstrip('.').strip() for caption in noisy_data_chunk]
                else:
                    delimiter = ""
                interval_text = delimiter.join(noisy_data_chunk)

                # Create prompt
                prompt = prompt_template.format(
                    interval_text=interval_text,
                    question=question,
                    **options,
                    length=len(noisy_data_chunk),
                    words=words
                )

                # Get completion
                summary = self.get_completion_from_text(
                    text=prompt,
                    completion_start=completion_start,
                    max_new_tokens=self.get_summary_from_noisy_perceptive_data_config.get("max_new_tokens"),
                    temperature=self.get_summary_from_noisy_perceptive_data_config.get("temperature")
                ).strip()

                if no_recursion:
                    # Remove ending period if exists and add interval format
                    if summary.endswith("."):
                        summary = summary[:-1]
                    summary = f"{summary} ({i * interval_span}-{(i + 1) * interval_span})."

                if remove_linebreaks:
                    summary = summary.replace('\n', ' ').strip()
                    # Replace multiple whitespaces with single space
                    summary = " ".join(summary.split())

                new_noisy_data.append(summary)

            noisy_data = new_noisy_data
            if no_recursion:
                break

        if not no_recursion:
            assert len(noisy_data) == 1, "Noisy data should be reduced to a single summary."
            summary = noisy_data[0]
        else:
            summary = " ".join(noisy_data)

        return summary

    def get_completion_from_text(
            self,
            text: str,
            completion_start: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> str:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        # Build the model if it should only be loaded when needed
        if self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built on demand.")

        # Get completion from the LLM
        completion = self.llm.get_completion(
            text=text,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        logger.debug(f"LLM completion: {completion}")

        # Destroy the model to free up resources if it was loaded on demand
        if self.load_models_only_when_needed:
            self.llm.destroy_model()
            logger.debug("LLM model destroyed after completion.")

        return completion

    def get_unspecific_objects_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        pre_inferred_path = self.get_unspecific_objects_from_video_clip_config.get("pre_inferred_object_detections_path")
        if pre_inferred_path:
            # Load pre-inferred object detections if a path is given in the config
            object_detections_data = read_json_file(file_path=pre_inferred_path)
            object_detections = object_detections_data.get(video_clip.id, [])

            logger.debug(f"Sampled indices: {video_clip.sampled_indices}.")
            logger.debug(f"Original FPS: {video_clip.original_fps}.")

            final_object_detections = get_clip_data_from_video_data(
                video_data=object_detections,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )
            logger.debug("Loaded pre-inferred unspecific object detections from path.")
        else:
            # Infer the object detections (this is time-consuming)
            start_frame = video_clip.sampled_indices[0]

            # Get unspecific objects for the video_clip_data using CogAgent
            objects_per_frame = infer_transcript_from_video_clip_using_frame_captions(
                video_clip=video_clip.data,
                start_frame=start_frame,
                fps=video_clip.original_fps,
                original_fps=video_clip.original_fps,
                frame_prompt=self.get_unspecific_objects_from_video_clip_config.get("frame_prompt", ""),
                model_id=self.get_unspecific_objects_from_video_clip_config.get("model_id", ""),
                tokenizer_id=self.get_unspecific_objects_from_video_clip_and_text_config.get("tokenizer_id", ""),
                device=self.get_unspecific_objects_from_video_clip_and_text_config.get("device", "cuda"),
                precision=self.get_unspecific_objects_from_video_clip_and_text_config.get("precision", "fp16"),
                quantization=self.get_unspecific_objects_from_video_clip_and_text_config.get("quantization", "int8"),
                temperature=self.get_unspecific_objects_from_video_clip_and_text_config.get("temperature", 0.7),
                max_new_tokens=self.get_unspecific_objects_from_video_clip_and_text_config.get("max_new_tokens", 50),
                do_sample=self.get_unspecific_objects_from_video_clip_and_text_config.get("do_sample", True)
            )
            logger.debug("Inferred unspecific object detections from video clip using CogAgent.")

            # Reduce to the very first action caption of each interval for now
            raw_object_detections = [captions.split(',')[0].strip() for captions in objects_per_frame.values()]
            raw_object_detections = dict(zip(objects_per_frame.keys(), raw_object_detections))

            # Use as many objects as specified in the config
            final_object_detections = []
            for interval, completion in raw_object_detections.items():
                # Assuming that we prompted the model to list objects (e.g., "provide an enumerated list")
                objects_in_completion = parse_list(completion)

                # Only use the specified number of objects per frame
                num_objects = self.get_unspecific_objects_from_video_clip_config.get('num_objects_per_frame', 1)
                assert num_objects > 0, "The number of objects per frame should be greater than 0 (num_objects_per_frame)."

                # Get the specified number of objects
                objects = objects_in_completion[:num_objects]

                # Clean up object names
                cleaned_objects = []
                for obj in objects:
                    obj = obj.rstrip('.').strip().rstrip(',').strip()
                    obj = obj.replace(", and", "").strip()
                    if len(obj) > 0:
                        cleaned_objects.append(obj)

                final_object_detections.extend(cleaned_objects)
                logger.debug(f"Processed objects for interval '{interval}': {cleaned_objects}")

            logger.debug("Final unspecific object detections extracted.")

        return final_object_detections

    def get_specific_object_confidences_from_video_clip(
            self,
            video_clip: VideoClip,
            question: str,
            options: Dict[str, str]
    ) -> Dict[str, float]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        if self.get_specific_objects_from_video_clip_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        # Format the prompt
        prompt_template = self.get_specific_objects_from_video_clip_config.get("prompt_template", "{}")
        prompt = prompt_template.format(
            question=question,
            option_0=options.get("option 0", ""),
            option_1=options.get("option 1", ""),
            option_2=options.get("option 2", ""),
            option_3=options.get("option 3", ""),
            option_4=options.get("option 4", "")
        )
        completion_start = self.get_specific_objects_from_video_clip_config.get("completion_start", "")

        # Get completion prediction from LLM
        prediction = self.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=self.get_specific_objects_from_video_clip_config.get("max_new_tokens"),
            temperature=self.get_specific_objects_from_video_clip_config.get("temperature")
        )
        logger.debug(f"Derived LLM completion about objects of the task to pay attention to while watching the video: {prediction}")

        # Parse and filter objects
        objects = filter_list_of_objects(parse_list(prediction))
        logger.debug(f"Parsed and filtered objects: {objects}")

        confidences_per_object = {}
        for obj in objects:
            object_detections = self.get_object_detections_from_video_clip_and_text(video_clip=video_clip, text=obj)
            # Placeholder for calculating mean confidence
            # Replace with actual confidence calculation based on object_detections
            mean_confidence = np.random.rand()  # Replace with actual calculation
            mean_confidence = 1 / (1 + np.exp(-10 * (mean_confidence - 0.3)))
            confidences_per_object[obj] = mean_confidence
            logger.debug(f"Object: {obj}, Mean Confidence: {mean_confidence}")

        return confidences_per_object
    #######################################################################################################################################
    #########################################################################################################################################
    
    ## Blanks Space Filling
    class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: Dict[str, Any],
            get_action_captions_from_video_clip_config: Dict[str, Any],
            get_temporal_grounding_from_video_clip_and_text_config: Dict[str, Any],
            get_summary_from_noisy_perceptive_data_config: Dict[str, Any],
            get_unspecific_objects_from_video_clip_config: Dict[str, Any],
            get_completion_from_text_config: Dict[str, Any],
            get_specific_objects_from_video_clip_config: Dict[str, Any],
            random_seed: int,
            reset_seed_for_each_function: bool = True,
            load_models_only_when_needed: bool = False
    ):
        # Initialize member variables
        self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
        self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
        self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
        self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config
        self.get_completion_from_text_config = get_completion_from_text_config
        self.get_unspecific_objects_from_video_clip_config = get_unspecific_objects_from_video_clip_config
        self.get_specific_objects_from_video_clip_config = get_specific_objects_from_video_clip_config
        self.load_models_only_when_needed = load_models_only_when_needed
        self.random_seed = random_seed
        self.reset_seed_for_each_function = reset_seed_for_each_function

        # Initialize LLM
        self._initialize_llm()

    def _initialize_llm(self):
        available_llm_classes = {
            "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        llm_class = self.get_completion_from_text_config.get("llm_class")
        if llm_class not in available_llm_classes:
            raise ValueError(f"Unsupported LLM class: {llm_class}")

        self.llm = available_llm_classes[llm_class](self.get_completion_from_text_config)

        # If the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built during initialization.")

    def reset_seed(self):
        # Clear CUDA cache and invalidate caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        importlib.invalidate_caches()
        logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Any:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        detections = infer_bounding_boxes_from_video(
            video_tensor=video_clip.data,
            obj=text,
            config_file=self.get_object_detections_from_video_clip_and_text_config["config_file"],
            checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
            box_threshold=self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
            text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
            cuda=self.get_object_detections_from_video_clip_and_text_config["cuda"]
        )

        logger.debug(f"Object detections for '{text}': {detections}")
        return detections

    def get_action_captions_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        final_action_captions = None
        model_name = self.get_action_captions_from_video_clip_config.get("model_name")

        if model_name not in ["LaViLa", "CogAgent"]:
            raise ValueError(f"Model name {model_name} is not supported for action captioning.")

        pre_inferred_path = self.get_action_captions_from_video_clip_config.get("pre_inferred_action_captions_path")
        if pre_inferred_path is not None:
            # Load pre-inferred action captions if a path is given in the config
            all_action_captions = read_json_file(file_path=pre_inferred_path)
            action_captions = all_action_captions.get(video_clip.id, {})
            final_action_captions = get_clip_data_from_video_data(
                video_data=action_captions,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )
            logger.warning("Using pre-inferred action captions. The pre-inferred action captions "
                           "should represent intervals of 1 second for each second of the video (180 captions for "
                           "EgoSchema).")
        else:
            if model_name == "LaViLa":
                # Infer action captions from the video clip using LaViLa
                sample_rate = self.get_action_captions_from_video_clip_config.get("resample_rate", 1)
                resampled_video_clip = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
                video_clip_data = resampled_video_clip.data
                start_frame = video_clip.sampled_indices[0]

                action_captions = infer_transcript_from_video_clip_using_action_captions(
                    video_clip=video_clip_data,
                    start_frame=start_frame,
                    fps=sample_rate,
                    original_fps=video_clip.original_fps,
                    interval_in_seconds=self.get_action_captions_from_video_clip_config.get("interval_in_seconds", 1),
                    temperature=self.get_action_captions_from_video_clip_config.get("temperature", 0.7),
                    top_p=self.get_action_captions_from_video_clip_config.get("top_p", 0.9),
                    max_text_length=self.get_action_captions_from_video_clip_config.get("max_new_tokens", 50),
                    num_return_sequences=self.get_action_captions_from_video_clip_config.get("num_return_sequences", 1),
                    early_stopping=self.get_action_captions_from_video_clip_config.get("early_stopping", True),
                    num_seg=self.get_action_captions_from_video_clip_config.get("num_seg", 4),
                    cuda=True,
                    modelzoo_dir_path=self.get_action_captions_from_video_clip_config.get("modelzoo_dir_path", ""),
                    checkpoint_download_url=self.get_action_captions_from_video_clip_config.get("checkpoint_download_url", ""),
                    checkpoint_file=self.get_action_captions_from_video_clip_config.get("checkpoint", "")
                )
            elif model_name == "CogAgent":
                # Infer action captions from the video clip using CogAgent
                start_frame = video_clip.sampled_indices[0]
                video_clip_data = video_clip.data

                action_captions = infer_transcript_from_video_clip_using_frame_captions(
                    video_clip=video_clip_data,
                    start_frame=start_frame,
                    fps=video_clip.original_fps,
                    original_fps=video_clip.original_fps,
                    frame_prompt=self.get_action_captions_from_video_clip_config.get("frame_prompt", ""),
                    model_id=self.get_action_captions_from_video_clip_config.get("model_id", ""),
                    tokenizer_id=self.get_action_captions_from_video_clip_config.get("tokenizer_id", ""),
                    device=self.get_action_captions_from_video_clip_config.get("device", "cuda"),
                    precision=self.get_action_captions_from_video_clip_config.get("precision", "fp16"),
                    quantization=self.get_action_captions_from_video_clip_config.get("quantization", "int8"),
                    temperature=self.get_action_captions_from_video_clip_config.get("temperature", 0.7),
                    max_new_tokens=self.get_action_captions_from_video_clip_config.get("max_new_tokens", 50),
                    do_sample=self.get_action_captions_from_video_clip_config.get("do_sample", True)
                )

            # Reduce to the very first action caption of each interval for now
            final_action_captions = [captions[0] for captions in action_captions.values()]
            logger.debug("Action captions generated and reduced to the first of each interval.")

        # Ensure final_action_captions is not None
        assert final_action_captions is not None, "Final action captions should not be None."
        return final_action_captions

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Dict[str, Any]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
            video=video_clip.data,
            text=text,
            config_dir=self.get_temporal_grounding_from_video_clip_and_text_config.get("config_dir", ""),
            checkpoint_path=self.get_temporal_grounding_from_video_clip_and_text_config.get("checkpoint", ""),
            clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config.get("clip_model_version", ""),
            output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config.get("output_feat_size", 512),
            half_precision=self.get_temporal_grounding_from_video_clip_and_text_config.get("half_precision", True),
            jit=self.get_temporal_grounding_from_video_clip_and_text_config.get("jit", False),
            resize_size=self.get_temporal_grounding_from_video_clip_and_text_config.get("resize_size", 224),
            gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config.get("gpu_id", 0)
        )

        foreground_indicators = text_temporal_grounding.get("foreground_indicators", [])
        boundary_offsets = text_temporal_grounding.get("boundary_offsets", [])
        saliency_scores = text_temporal_grounding.get("saliency_scores", [])

        # Derive the best k boundary offset indices
        top_k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        top_k_indices = sorted(range(len(foreground_indicators)), key=lambda i: foreground_indicators[i], reverse=True)[:top_k]

        # Initialize relevance_indicators with 0 for each frame in video_clip
        relevance_indicators = [0] * len(video_clip.sampled_indices)

        for idx in top_k_indices:
            boundary_offset = boundary_offsets[idx]
            start_offset, end_offset = boundary_offset

            # Optimistic flooring and ceiling
            start_index = max(0, int(start_offset * len(video_clip.sampled_indices)))
            end_index = min(int(end_offset * len(video_clip.sampled_indices)), len(video_clip.sampled_indices) - 1)

            # Update relevance indicators
            for i in range(start_index, end_index + 1):
                relevance_indicators[i] = 1

        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        # Handle saliency scores
        salience_indicators = []
        if isinstance(saliency_scores, torch.Tensor):
            num_saliency_scores = saliency_scores.size(0) if saliency_scores.dim() > 0 else 1
            for i in range(num_saliency_scores):
                saliency_score = saliency_scores[i].item() if num_saliency_scores > 1 else saliency_scores.item()
                saliency_score = max(0.0, min(1.0, saliency_score))
                salience_indicators.append(saliency_score)
        else:
            # If saliency_scores is not a tensor, assume it's a list
            salience_indicators = [max(0.0, min(1.0, score)) for score in saliency_scores]

        logger.debug("Derived salience indicators.")

        return {
            "foreground_indicators": foreground_indicators,
            "relevance_indicators": relevance_indicators,
            "salience_indicators": salience_indicators
        }

    def get_summary_from_noisy_perceptive_data(
            self,
            action_captions: List[str],
            object_detections: List[str],
            temporal_grounding: List[str],
            interleaved_action_captions_and_object_detections: List[str],
            video_clip: VideoClip,
            question: Optional[str] = "",
            options: Optional[Dict[str, str]] = None,
            words: int = 500
    ) -> str:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        pre_inferred_summaries_path = self.get_summary_from_noisy_perceptive_data_config.get("pre_inferred_summaries_path")
        if pre_inferred_summaries_path is not None:
            logger.debug("Using pre-inferred summaries.")

            # Load pre-inferred summaries JSON file
            summaries = read_json_file(file_path=pre_inferred_summaries_path)

            # Get the summaries entry for the video clip
            video_summaries = summaries.get(video_clip.id, {})
            logger.debug(f"Loaded summaries for video_clip ID {video_clip.id}.")

            # Get the frame interval of the clip
            sampled_indices = video_clip.sampled_indices
            start_frame = sampled_indices[0]
            end_frame = sampled_indices[-1]

            # Define a tolerance threshold for the frame interval
            tolerance = video_clip.original_fps // 2

            # Define the whole video start and end frame
            whole_video_start_frame = 0
            whole_video_end_frame = video_clip.original_num_frames - 1

            # If the video clip covers the whole video, try to take the whole video summary first
            if start_frame == whole_video_start_frame and end_frame == whole_video_end_frame:
                try:
                    if len(action_captions) != 0:
                        return video_summaries["root_action_captions_summaries"]
                    elif len(object_detections) != 0:
                        return video_summaries["root_object_detections_summaries"]
                    elif len(temporal_grounding) != 0:
                        return video_summaries["root_temporal_grounding_summaries"]
                    else:
                        raise ValueError("Please provide action captions, object detections, or temporal grounding to summarize.")
                except KeyError:
                    logger.warning("No whole video summary found. Trying to find the summary "
                                   "for the clip as long as the video.")

            # Look for the summaries of the clip in the pre-inferred summaries
            index = None
            for i, clip_boundary in enumerate(video_summaries.get("clip_boundaries", [])):
                clip_start, clip_end = clip_boundary
                if abs(clip_start - start_frame) <= tolerance and abs(clip_end - end_frame) <= tolerance:
                    index = i
                    break

            if index is None:
                raise ValueError("No matching clip boundary found within tolerance.")

            if len(action_captions) != 0:
                summary_type = "action_caption_summaries"
                fallback_summary_type = "n.a."
            elif len(object_detections) != 0:
                summary_type = "object_detections_summaries"
                fallback_summary_type = "unspecific_object_detection_summaries"
            elif len(temporal_grounding) != 0:
                summary_type = "temporal_grounding_summaries"
                fallback_summary_type = "n.a."
            else:
                raise ValueError("No valid data provided for summary.")

            summary = video_summaries.get(summary_type, video_summaries.get(fallback_summary_type, ""))
            return summary

        # If pre-inferred summaries are not available, generate them
        prompt_template = self.get_summary_from_noisy_perceptive_data_config.get("prompt_template", "{}")
        completion_start = self.get_summary_from_noisy_perceptive_data_config.get("completion_start", "")
        interval_span = self.get_summary_from_noisy_perceptive_data_config.get("interval_span", 10)
        no_recursion = self.get_summary_from_noisy_perceptive_data_config.get("no_recursion", False)
        remove_linebreaks = self.get_summary_from_noisy_perceptive_data_config.get("remove_linebreaks", False)

        if options is None:
            options = {f"option_{i}": f"Option {i}" for i in range(4)}

        if question and self.get_summary_from_noisy_perceptive_data_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)

        def chunks(data: List[str], chunk_size: int):
            for j in range(0, len(data), chunk_size):
                yield data[j:j + chunk_size]

        noisy_data = action_captions + object_detections + temporal_grounding + interleaved_action_captions_and_object_detections

        while len(noisy_data) > 1:
            new_noisy_data = []
            for i, noisy_data_chunk in enumerate(chunks(noisy_data, chunk_size=interval_span)):
                # Remove trailing dots if necessary
                if not temporal_grounding:
                    delimiter = ". "
                    noisy_data_chunk = [caption.rstrip('.').strip() for caption in noisy_data_chunk]
                else:
                    delimiter = ""
                interval_text = delimiter.join(noisy_data_chunk)

                logger.debug(f"Noisy Data Chunk: {noisy_data_chunk}")
                logger.debug(f"Interval text: {interval_text}")

                # Create prompt
                prompt = prompt_template.format(
                    interval_text=interval_text,
                    question=question,
                    **options,
                    length=len(noisy_data_chunk),
                    words=words
                )
                logger.debug(f"Noisy Data Summarization Prompt: {prompt}")

                # Get completion
                summary = self.get_completion_from_text(
                    text=prompt,
                    completion_start=completion_start,
                    max_new_tokens=self.get_summary_from_noisy_perceptive_data_config.get("max_new_tokens"),
                    temperature=self.get_summary_from_noisy_perceptive_data_config.get("temperature")
                ).strip()

                if no_recursion:
                    # Remove ending period if exists and add interval format
                    if summary.endswith("."):
                        summary = summary[:-1]
                    summary = f"{summary} ({i * interval_span}-{(i + 1) * interval_span})."

                if remove_linebreaks:
                    summary = summary.replace('\n', ' ').strip()
                    # Replace multiple whitespaces with single space
                    summary = " ".join(summary.split())

                # Add summary to new_noisy_data
                new_noisy_data.append(summary)

            # Update noisy_data with new_noisy_data
            noisy_data = new_noisy_data
            if no_recursion:
                break

        if not no_recursion:
            assert len(noisy_data) == 1, "Noisy data should be reduced to a single summary."
            summary = noisy_data[0]
        else:
            summary = " ".join(noisy_data)

        return summary

    def get_completion_from_text(
            self,
            text: str,
            completion_start: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> str:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        # Build the model if it should only be loaded when needed
        if self.load_models_only_when_needed:
            self.llm.build_model()
            logger.debug("LLM model built on demand.")

        # Get completion from the LLM
        completion = self.llm.get_completion(
            text=text,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        logger.debug(f"LLM completion: {completion}")

        # Destroy the model to free up resources if it was loaded on demand
        if self.load_models_only_when_needed:
            self.llm.destroy_model()
            logger.debug("LLM model destroyed after completion.")

        return completion

    def get_unspecific_objects_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        pre_inferred_path = self.get_unspecific_objects_from_video_clip_config.get("pre_inferred_object_detections_path")
        if pre_inferred_path:
            # Load pre-inferred object detections if a path is given in the config
            object_detections_data = read_json_file(file_path=pre_inferred_path)
            object_detections = object_detections_data.get(video_clip.id, [])

            logger.debug(f"Sampled indices: {video_clip.sampled_indices}.")
            logger.debug(f"Original FPS: {video_clip.original_fps}.")

            final_object_detections = get_clip_data_from_video_data(
                video_data=object_detections,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )
            logger.debug("Loaded pre-inferred unspecific object detections from path.")
        else:
            # Infer the object detections (this is time-consuming)
            start_frame = video_clip.sampled_indices[0]

            # Get unspecific objects for the video_clip_data using CogAgent
            objects_per_frame = infer_transcript_from_video_clip_using_frame_captions(
                video_clip=video_clip.data,
                start_frame=start_frame,
                fps=video_clip.original_fps,
                original_fps=video_clip.original_fps,
                frame_prompt=self.get_unspecific_objects_from_video_clip_config.get("frame_prompt", ""),
                model_id=self.get_unspecific_objects_from_video_clip_config.get("model_id", ""),
                tokenizer_id=self.get_unspecific_objects_from_video_clip_config.get("tokenizer_id", ""),
                device=self.get_unspecific_objects_from_video_clip_config.get("device", "cuda"),
                precision=self.get_unspecific_objects_from_video_clip_config.get("precision", "fp16"),
                quantization=self.get_unspecific_objects_from_video_clip_config.get("quantization", "int8"),
                temperature=self.get_unspecific_objects_from_video_clip_config.get("temperature", 0.7),
                max_new_tokens=self.get_unspecific_objects_from_video_clip_config.get("max_new_tokens", 50),
                do_sample=self.get_unspecific_objects_from_video_clip_config.get("do_sample", True)
            )
            logger.debug("Inferred unspecific object detections from video clip using CogAgent.")

            # Reduce to the very first action caption of each interval for now
            raw_object_detections = [captions.split(',')[0].strip() for captions in objects_per_frame.values()]
            raw_object_detections = dict(zip(objects_per_frame.keys(), raw_object_detections))

            # Use as many objects as specified in the config
            final_object_detections = []
            for interval, completion in raw_object_detections.items():
                # Assuming that we prompted the model to list objects (e.g., "provide an enumerated list")
                objects_in_completion = parse_list(completion)

                # Only use the specified number of objects per frame
                num_objects = self.get_unspecific_objects_from_video_clip_config.get('num_objects_per_frame', 1)
                assert num_objects > 0, "The number of objects per frame should be greater than 0 (num_objects_per_frame)."

                # Get the specified number of objects
                objects = objects_in_completion[:num_objects]

                # Clean up object names
                cleaned_objects = []
                for obj in objects:
                    obj = obj.rstrip('.').strip().rstrip(',').strip()
                    obj = obj.replace(", and", "").strip()
                    if len(obj) > 0:
                        cleaned_objects.append(obj)

                final_object_detections.extend(cleaned_objects)
                logger.debug(f"Processed objects for interval '{interval}': {cleaned_objects}")

            logger.debug("Final unspecific object detections extracted.")

        return final_object_detections

    def get_specific_object_confidences_from_video_clip(
            self,
            video_clip: VideoClip,
            question: str,
            options: Dict[str, str]
    ) -> Dict[str, float]:
        # Reset the seed to ensure reproducibility
        self.reset_seed()

        if self.get_specific_objects_from_video_clip_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        # Format the prompt
        prompt_template = self.get_specific_objects_from_video_clip_config.get("prompt_template", "{}")
        prompt = prompt_template.format(
            question=question,
            option_0=options.get("option 0", ""),
            option_1=options.get("option 1", ""),
            option_2=options.get("option 2", ""),
            option_3=options.get("option 3", ""),
            option_4=options.get("option 4", "")
        )
        completion_start = self.get_specific_objects_from_video_clip_config.get("completion_start", "")

        # Get completion prediction from LLM
        prediction = self.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=self.get_specific_objects_from_video_clip_config.get("max_new_tokens"),
            temperature=self.get_specific_objects_from_video_clip_config.get("temperature")
        )
        logger.debug(f"Derived LLM completion about objects of the task to pay attention to while watching the video: {prediction}")

        # Parse and filter objects
        objects = filter_list_of_objects(parse_list(prediction))
        logger.debug(f"Parsed and filtered objects: {objects}")

        confidences_per_object = {}
        for obj in objects:
            # Infer the object detections of the object in the video clip
            object_detections = self.get_object_detections_from_video_clip_and_text(video_clip=video_clip, text=obj)

            # Calculate mean confidence of the object detection
            if not object_detections.get("detections"):
                mean_confidence = 0.0
            else:
                mean_confidence = sum([det.get("probability", 0.0) for det in object_detections["detections"]]) / len(object_detections["detections"])

            # Apply sigmoid function with inflection point at 0.3
            mean_confidence = 1 / (1 + np.exp(-10 * (mean_confidence - 0.3)))

            confidences_per_object[obj] = mean_confidence
            logger.debug(f"Object: {obj}, Mean Confidence: {mean_confidence}")

        return confidences_per_object

