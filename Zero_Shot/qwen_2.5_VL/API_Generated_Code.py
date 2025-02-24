# #######################################################
# #######################################################
# Based on Description to generate code
# #######################################################
# #######################################################
class API:
    def __init__(self,
                 get_object_detections_from_video_clip_and_text_config: dict[str, any],
                 get_action_captions_from_video_clip_config: dict[str, any],
                 get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
                 get_summary_from_noisy_perceptive_data_config: dict[str, any],
                 get_unspecific_objects_from_video_clip_config: dict[str, any],
                 get_completion_from_text_config: dict[str, any],
                 get_specific_objects_from_video_clip_config: dict[str, any],
                 random_seed: int,
                 reset_seed_for_each_function: bool = True,
                 load_models_only_when_needed: bool = False):
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
        
    def _initialize_llm(self) -> None:
        """Initialize LLM based on configuration."""
        available_llm_classes = {
            'HuggingFaceLLM': lambda x: HuggingFaceLLM.initialize_huggingface_llm_from_config(x),
            'OpenAILLM': lambda x: OpenAILLM.initialize_openai_llm_from_config(x)
        }
        llm_class_key = self.get_completion_from_text_config['llm_class']
        try:
            self.llm = available_llm_classes[llm_class_key](self.get_completion_from_text_config)
        except KeyError:
            raise ValueError(f"Invalid LLMSpecification '{llm_class_key}'")
            
        if not self.load_models_only_when_needed:
            self.llm.build_model()
    
    def reset_seed(self) -> None:
        """Reset seeds before running operations that depend on randomness"""
        import torch
        import numpy as np
        print('Random seed is being cleared.')
        torch.cuda.empty_cache()
        if not self.reset_seed_for_each_function:
            print('Random seed is not reset for each function call')
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip) -> Any:
        """
        Get object detections from video clip using provided configurations.
        """
        self.reset_seed()
        return infer_bounding_boxes_from_video(video_clip=video_clip)

    def get_action_captions_from_video_clip(self, video_clip: VideoClip) -> List[Any]:
        """
        Generate action captions from video clips according to specified configurations.
        """
        self.reset_seed()
        final_action_captions = None
        model_name = self.get_action_captions_from_video_clip_config['model_name']

        if model_name != "LaViLa" and model_name != "CogAgent":
            raise ValueError(f"Model name must be either LaViLa or CogAgent but got {model_name}")

        if (path := self.get_action_captions_from_video_clip_config['pre_inferred_action_captions_path']) is not None:
            final_action_captions = read_json_file(path)

        if model_name == "LaViLa":
            resample_rate = self.get_action_captions_from_video_clip_config['resample_rate']
            video_clip_data, start_frame = ..., ...  # Placeholder values needed during actual implementation
            
            transcript_func = getattr(lavila_video_captioner.narrator_inference, f"infer_{model_name}_transcript_from_video_clip")(
                video_clip_data, start_frame=start_frame, resample_rate=resample_rate
            )
            action_captions = transcript_func(infervideoclip=..., ...)  # Placeholder parameters required
                
        elif model_name == "CogAgent":
            video_clip_data, start_frame = ..., ...
            transcript_func = getattr(CogAgent.video, f"infer_{model_name}_transcript_from_video_clip")(video_clip_data, start_frame=start_frame)
            action_captions = transcript_func(...)  # Placeholder parameters required

        reduced_action_captions = [action_captions[i]['caption'] for i in range(len(action_captions))]
        assert len(reduced_action_captions), "Action Captions cannot be empty"
        return reduced_action_captions
    
    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str) -> Dict[str, Any]:
        """
        Perform temporal grounding analysis over video data and textual information.
        """
        self.reset_seed()
        text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(text=text, video_clip=video_clip)
        fgd = text_temporal_grounding['foreground_indicators']
        bo = text_temporal_grounding['boundary_offsets']
        ss = text_temporal_grounding['saliency_scores']

        fgd_list = []
        for fgds in fgd:
            fgd_list.extend([fgds])

        top_k_indices = sorted(range(len(bo)), reverse=True)[:int(self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals"))]
        top_k_fgds = [fgd[top_k_indices[i]] for i in range(int(self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals")))]

        relevance_indicators = [0]*len(video_clip)
        for idx in top_k_fgds:
            start_idx = max(idx[0]-5, 0)
            end_idx = min(idx[1]+6, len(video_clip))
            relevance_indicators[start_idx:end_idx+1] = [1]*(end_idx-start_idx+1)

        salience_indicators = []

        if ss.dim() > 0:
            num_saliency_scores = ss.size(0)
            for ssi in range(num_saliency_scores):
                saliency_score = max(min(ss[ssi]), 0.)
                salience_indicators.append(saliency_score)

        return {"foreground_indicators": fgd_list[:num_saliency_scores],
                "relevance_indicators": relevance_indicators,
                "salience_indicators": salience_indicators}
        
# #######################################################
# #######################################################
# Completion given Half of the code
#######################################################
#######################################################

# Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the above requirment and description.
# \n\n class API:
#     def __init__(
#             self,
#             get_object_detections_from_video_clip_and_text_config: dict[str, any],
#             get_action_captions_from_video_clip_config: dict[str, any],
#             get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
#             get_summary_from_noisy_perceptive_data_config: dict[str, any],
#             get_unspecific_objects_from_video_clip_config: dict[str, any],
#             get_completion_from_text_config: dict[str, any],
#             get_specific_objects_from_video_clip_config: dict[str, any],
#             random_seed: int,
#             reset_seed_for_each_function: bool = True,
#             load_models_only_when_needed: bool = False
#     ):
#         self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
#         self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
#         self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
#         self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config

#         def _initialize_llm(self):

#         available_llm_classes = {
#             "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
        
#           def reset_seed(self):
#         # clear caches
#         torch.cuda.empty_cache()
#         importlib.invalidate_caches()
#         logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

#         if not self.reset_seed_for_each_function:
#             logger.info("Random seed is not reset for each function call.")
#             return
        
#          def get_action_captions_from_video_clip(self, video_clip: VideoClip):
#         # TODO do not use dict, only use list as return type

#         # reset the seed to ensure reproducibility
#         self.reset_seed()

#         final_action_captions = None
#         if self.get_action_captions_from_video_clip_config["model_name"] not in ["LaViLa", "CogAgent"]:
#             raise ValueError(f"Model name {self.get_action_captions_from_video_clip_config['model_name']} "
#                              f"is not supported for action captioning.")

#            def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
#         # reset the seed to ensure reproducibility
#         self.reset_seed()

#         text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
#             video=video_clip.data,
#             text=text,
#             config_dir=self.get_temporal_grounding_from_video_clip_and_text_config["config_dir"],
#             checkpoint_path=self.get_temporal_grounding_from_video_clip_and_text_config["checkpoint"],
#             clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config["clip_model_version"],
#             output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config["output_feat_size"],
#             half_precision=self.get_temporal_grounding_from_video_clip_and_text_config["half_precision"],
#             jit=self.get_temporal_grounding_from_video_clip_and_text_config["jit"],
#             resize_size=self.get_temporal_grounding_from_video_clip_and_text_config["resize"],
#             gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config["gpu_id"]
#         )

#         foreground_indicators = text_temporal_grounding["foreground_indicators"]
#         boundary_offsets = text_temporal_grounding["boundary_offsets"]
# #######################################################
# #######################################################

class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: dict[str, any],
            get_action_captions_from_video_clip_config: dict[str, any],
            get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
            get_summary_from_noisy_perceptive_data_config: dict[str, any],
            get_unspecific_objects_from_video_clip_config: dict[str, any],
            get_completion_from_text_config: dict[str, any],
            get_specific_objects_from_video_clip_config: dict[str, any],
            random_seed: int,
            reset_seed_for_each_function: bool = True,
            load_models_only_when_needed: bool = False
    ):
        self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
        self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
        self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
        self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config

        def _initialize_llm(self):
            """
            Initializes LLMs according to configuration settings.
            """
            available_llm_classes = {
                "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
                "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config,
            }
            
            llm_class_key = self.get_completion_from_text_config.get('llm_class')
            if llm_class_key is None:
                raise ValueError("No valid LLMS specified")
                
            try:
                self.llm = available_llm_classes[llm_class_key](self.get_completion_from_text_config)
            except KeyError:
                raise ValueError(f"No such model '{llm_class_key}' found")

            if not self.load_models_only_when_needed:
                self.llm.build_model()

        def reset_seed(self):
            """Clears CUDA cache and sets seeds."""
            torch.cuda.empty_cache()
            importlib.invalidate_caches()
            logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

            if not self.reset_seed_for_each_function:
                logger.info("Random seed is not reset for each function call.")
                return
            
            np.random.seed(random_seed)  
            torch.manual_seed(random_seed)

        def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip):
            """Detect objects within videos using provided configurations"""
            self.reset_seed()
            return infer_bounding_boxes_from_video(video_clip)

        def get_action_captions_from_video_clip(self, video_clip: VideoClip):
            """Generate action captions from video clips"""

            self.reset_seed()
            final_action_captions = None
            if self.get_action_captions_from_video_clip_config["model_name"] not in ["LaViLa", "CogAgent"]:
                raise ValueError(f"Model name {self.get_action_captions_from_video_clip_config['model_name']} "
                                 f"is not supported for action captioning.")

            if (path := self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"]) is not None:
                final_action_captions = read_json_file(path)

            elif self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa":
                resample_rate = self.get_action_captions_from_video_clip_config["resample_rate"]

                video_clip_data, start_frame = get_video_clip_data_and_start_frame(video_clip)
                if resample_rate > 0:
                    video_clip_data.resample(resample_rate)

                action_captions = infer_transcript_from_video_clip_using_action_captions(video_clip_data, start_frame=start_frame)
                final_action_captions = action_captions[:]

            elif self.get_action_captions_from_video_clip_config["model_name"] == "CogAgent":
                cog_agent = CogAgent(video_clip)
                action_captions = infer_transcript_from_video_clip_using_frame_captions(cog_agent)
                final_action_captions = action_captions[:]

            assert final_action_captions is not None, "Action Captions cannot be null"
            return [caption.value for caption in final_action_captions]

        def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
            """Get temporal grounding scores from video data and textual information."""

            self.reset_seed()
            text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
                video=video_clip.data,
                text=text,
                config_dir=self.get_temporal_grounding_from_video_clip_and_text_config["config_dir"],
                checkpoint_path=self.get_temporal_grounding_from_video_clip_and_text_config["checkpoint"],
                clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config["clip_model_version"],
                output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config["output_feat_size"],
                half_precision=self.get_temporal_grounding_from_video_clip_and_text_config["half_precision"],
                jit=self.get_temporal_grounding_from_video_clip_and_text_config["jit"],
                resize_size=self.get_temporal_grounding_from_video_clip_and_text_config["resize"],
                gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config["gpu_id"]
            )
            
            foreground_indicators = text_temporal_grounding["foreground_indicators"]
            boundary_offsets = text_temporal_grounding["boundary_offsets"]
            saliency_scores = text_temporal_grounding["saliency_scores"]

            relevant_foreground_indicators = []
            top_k_indices = sorted(range(len(boundary_offsets)), reverse=True)[:min(k, len(boundary_offsets))]
            top_k_indices.sort(reverse=False)
            flattened_top_k_indices = sum([range(i * k + b - k // 2, i * k + b + k // 2 + 1) for i, b in enumerate(top_k_indices)], [])
            relevant_foreground_indicators.extend(flattened_top_k_indices)

            relevance_indicators = [0.] * len(relevant_foreground_indicators)
            current_idx = 0
            while current_idx < len(relevant_foreground_indicators):
                start_index = max(int(np.floor(relevant_foreground_indicators[current_idx])), 0)
                end_index = min(int(np.ceil(relevant_foreground_indicators[current_idx + 1]) - 1), len(video_clip))
                relevance_indicators[start_index:end_index+1] = [1]*len(range(start_index,end_index+1))

                debug_message = ("Relevance Indicators:\n" +
                                "\tStart Index: {}\n" +
                                "\tEnd Index: {}").format(start_index, end_index)
                logger.debug(debug_message)

                current_idx += 2

            salience_indicators = []

            if saliency_scores.numel():
                num_saliency_scores = saliency_scores.shape[-1]
                for idx in range(num_saliency_scores):
                    saliency_score = saliency_scores[idx].item()
                    normalized_score = max(min(saliency_score, 1.), 0.)
                    salience_indicators.append(normalized_score)

            return {"foreground_indicators": relevant_foreground_indicators,
                    "relevance_indicators": relevance_indicators.tolist(),
                    "salience_indicators": salience_indicators}





# #######################################################
# #######################################################
# Filling blanks in the code

# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description.

# \n\n class API:
#     def __init__(
#             self,
#             get_object_detections_from_video_clip_and_text_config: dict[str, any],
#             get_action_captions_from_video_clip_config: dict[str, any],
#             get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
#             get_summary_from_noisy_perceptive_data_config: dict[str, any],
#             : dict[str, any],
#             get_completion_from_text_config: dict[str, any],
#             get_specific_objects_from_video_clip_config: ,
#             random_seed: int,
#             reset_seed_for_each_function: bool = True,
#             load_models_only_when_needed: bool = False
#     ):
#         self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
#         self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
#         self. = get_temporal_grounding_from_video_clip_and_text_config
#         self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config
#         self.get_completion_from_text_config = get_completion_from_text_config
#         self.get_unspecific_objects_from_video_clip_config = 
#         self. = get_specific_objects_from_video_clip_config
#         self.load_models_only_when_needed = load_models_only_when_needed
#         self.random_seed = random_seed
#         self.reset_seed_for_each_function = 

#         self._initialize_llm()

#     def _initialize_llm(self):

#         available_llm_classes = {
#             "HuggingFaceLLM": .initialize_huggingface_llm_from_config,
#             "OpenAILLM": OpenAILLM.
#         }

#          = available_llm_classes[
#             self.get_completion_from_text_config["llm_class"]](
#             self.
#         )

#         # if the models should only be loaded when needed, do not build them now
#         if not self.load_models_only_when_needed:
#             self.llm.build_model()

#     def reset_seed(self):
#         # clear caches
#         torch.cuda.empty_cache()
#         importlib.invalidate_caches()
#         logger.info("")

#         if not self.reset_seed_for_each_function:
#             logger.info("Random seed is not reset for each function call.")
#             return

#         .seed(self.random_seed)
#         np.random.seed(self.random_seed)
#         torch.manual_seed(self.)
#         .cuda.manual_seed(self.random_seed)
#         logger.info(f"Random seed has been reset to {self.} to ensure reproducibility and comparability.")

#     def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
#         # reset the seed to ensure reproducibility
#         self.reset_seed()

#         return (
#             video_tensor=video_clip.data,
#             obj=text,
#             config_file=self.["config_file"],
#             checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
#             =self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
#             text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
#             cuda=self.["cuda"]
#         )

#     def get_action_captions_from_video_clip(self, video_clip: VideoClip):
#         # TODO do not use dict, only use list as return type

#         # reset the seed to ensure reproducibility
#         self.reset_seed()

#         final_action_captions = None
#         if self.get_action_captions_from_video_clip_config["model_name"] not in ["LaViLa", "CogAgent"]:
#             raise ValueError(f"Model name {self.get_action_captions_from_video_clip_config['model_name']} "
#                              f"is not supported for action captioning.")

#         if self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"] is not None:

#             # load pre-inferred action captions if a path for them is given in the config
#              = read_json_file(
#                 file_path=self.["pre_inferred_action_captions_path"])
#              = all_action_captions[video_clip.id]

#             final_action_captions = get_clip_data_from_video_data(
#                 video_data=action_captions,
#                 sampled_indices=video_clip.sampled_indices,
#                 fps=video_clip.
#             )

#             logger.warning("Using pre-inferred action captions. The pre-inferred action captions "
#                            "should represent intervals of 1 second for each second of the video (180 captions for"
#                            "EgoSchema).")

#         elif self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa":

#             # infer action captions from the video clip using LaViLa
#             sample_rate = self.["resample_rate"]

#             # get the full video tensor data
#              = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
#             video_clip_data = resampled_video_clip.data
#             start_frame = video_clip.[0]

#             # get captions for the video_clip_data using LaViLa
#             action_captions = infer_transcript_from_video_clip_using_action_captions(
#                 =video_clip_data,
#                 start_frame=start_frame,
#                 fps=sample_rate,
#                 original_fps=video_clip.original_fps,
#                 interval_in_seconds=self.["interval_in_seconds"],
#                 =self.get_action_captions_from_video_clip_config["temperature"],
#                 top_p=self.get_action_captions_from_video_clip_config["top_p"],
#                 max_text_length=self.get_action_captions_from_video_clip_config["max_new_tokens"],
#                 num_return_sequences=self.["num_return_sequences"],
#                 early_stopping=self.get_action_captions_from_video_clip_config["early_stopping"],
#                 =self.["num_seg"],
#                 cuda=True,
#                 modelzoo_dir_path=self.get_action_captions_from_video_clip_config["modelzoo_dir_path"],
#                 =self.get_action_captions_from_video_clip_config["checkpoint_download_url"],
#                 checkpoint_file=self.["checkpoint"]
#             )

#             # reduce to the very first action caption of each interval for now
#             # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
#              = [action_captions[0] for action_captions in action_captions.values()]
#             final_action_captions = dict(zip(.keys(), ))

#         elif self.get_action_captions_from_video_clip_config["model_name"] == "CogAgent":

#             start_frame = video_clip.sampled_indices[0]

#             # get captions for the video_clip_data using CogAgent
#             action_captions = infer_transcript_from_video_clip_using_frame_captions(
#                 =video_clip.data,
#                 start_frame=start_frame,
#                 original_fps=video_clip.,
#                 =self.get_action_captions_from_video_clip_config["frame_prompt"],
#                 model_id=self.["model_id"],
#                 tokenizer_id=self.get_action_captions_from_video_clip_config["tokenizer_id"],
#                 =self.get_action_captions_from_video_clip_config["device"],
#                 precision=self.get_action_captions_from_video_clip_config["precision"],
#                 quantization=self.["quantization"],
#                 =self.get_action_captions_from_video_clip_config["temperature"],
#                 max_new_tokens=self.["max_new_tokens"],
#                 =self.get_action_captions_from_video_clip_config["do_sample"]
#             )

#             # reduce to the very first action caption of each interval for now
#             # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
#             final_action_captions = [action_captions[0] for  in action_captions.values()]
#              = dict(zip(action_captions.keys(), final_action_captions))

#         assert final_action_captions is not None, "Action captions should have been inferred."
#         return list(.values())

#     def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
#         # reset the seed to ensure reproducibility
#         self.reset_seed()

#          = infer_temporal_grounding_score_from_video_and_text(
#             video=video_clip.data,
#             text=text,
#             config_dir=self.["config_dir"],
#             =self.get_temporal_grounding_from_video_clip_and_text_config["checkpoint"],
#             clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config["clip_model_version"],
#             output_feat_size=self.["output_feat_size"],
#             half_precision=self.get_temporal_grounding_from_video_clip_and_text_config["half_precision"],
#             jit=self.get_temporal_grounding_from_video_clip_and_text_config["jit"],
#             resize_size=self.get_temporal_grounding_from_video_clip_and_text_config["resize"],
#             gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config["gpu_id"]
#         )

#          = text_temporal_grounding["foreground_indicators"]
#         boundary_offsets = text_temporal_grounding["boundary_offsets"]
#          = text_temporal_grounding[""].squeeze()

#         # make a list of the foreground indicators
#          = [foreground_indicators[i].item() for i in range(foreground_indicators.(0))]
#         logger.debug("Derived foreground indicators")

#         # derive the best boundary offset indices
#         k = self.get_temporal_grounding_from_video_clip_and_text_config.get("", 1)
        
#         _, top_k_indices = torch.topk(.flatten(), k=k)

#         top_k_indices = top_k_indices.tolist()


#         # initialize the relevance indicators with zeros
#         relevance_indicators = [0 for _ in range(len(video_clip))]

#         # iteratively update the relevance indicators for the top k intervals
#         for top_i_index in top_k_indices:
#              = boundary_offsets[top_i_index].tolist()
#             logger.debug(f"Top {top_i_index} boundary offset: {top_i_boundary_offset}.")

#             # optimistic flooring of start index
#              = max(0, top_i_index + math.floor(top_i_boundary_offset[0] * len(video_clip)))
#             logger.debug(f"Start index: {start_index}.")

#             # optimistic ceiling of end index
#             end_index = min( + .ceil([1] * len(video_clip)), len() - 1)
#             logger.debug(f"End index: {end_index}.")

#             # update the relevance indicators
#             # i.e., set all relevance indicators between start and end index to 1
#             relevance_indicators = [1 if  <= i <= end_index else relevance_indicators[i] for i in
#                                     range(len(video_clip))]
#             logger.debug(f"Relevance indicators: {}.")
#         logger.debug(f"Derived relevance indicators: {relevance_indicators}")

#         salience_indicators = []
#         num_saliency_scores = 1 if saliency_scores.dim() == 0 else saliency_scores.size(0)
#         for i in range(num_saliency_scores):
#             saliency_score = [i].item() if num_saliency_scores > 1 else saliency_scores.item()
#             saliency_score = max(0.0, min(, saliency_score))
#             .append(saliency_score)
#         logger.debug("Derived salience indicators")

#         return {
#             "foreground_indicators": ,
#             "relevance_indicators": relevance_indicators,
#             "": salience_indicators
#         }
# #######################################################
# #######################################################
class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: dict[str, any],
            get_action_captions_from_video_clip_config: dict[str, any],
            get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
            get_summary_from_noisy_perceptive_data_config: dict[str, any],
            get_unspecific_objects_from_video_clip_config: dict[str, any],
            get_completion_from_text_config: dict[str, any],
            get_specific_objects_from_video_clip_config: dict[str, any],
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
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config,
        }

        self.llm = available_llm_classes[self.get_completion_from_text_config["llm_class"]](self.get_completion_from_text_config)

        # if the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            self.llm.build_model()

    def reset_seed(self):
        # clear caches
        torch.cuda.empty_cache()
        importlib.invalidate_caches()
        logger.info("Cleared CUDA and other caches.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        return (
            video_tensor=video_clip.data,
            obj=text,
            config_file=self.get_object_detections_from_video_clip_and_text_config["config_file"],
            checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
            box_threshold=self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
            text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
            cuda=self.get_object_detections_from_video_clip_and_text_config["cuda"]
        )

    def get_action_captions_from_video_clip(self, video_clip: VideoClip):
        # TODO do not use dict, only use list as return type

        # reset the seed to ensure reproducibility
        self.reset_seed()

        final_action_captions = None
        if self.get_action_captions_from_video_clip_config["model_name"] not in ["LaViLa", "CogAgent"]:
            raise ValueError(f"Model name {self.get_action_captions_from_video_clip_config['model_name']} "
                             f"is not supported for action captioning.")

        if self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"] is not None:

            # load pre-inferred action captions if a path for them is given in the config
            try:
                with open(self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"], 'r') as json_file:
                    all_action_captions = json.load(json_file)
            except FileNotFoundError:
                logging.error(f"The specified JSON file at '{self.get_action_captions_from_video_clip_config['pre_inferred_action_captions_path']}' does not exist.")
                exit(-1)
            
            final_action_captions = get_clip_data_from_video_data(
                video_data=all_action_captions[video_clip.id],
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.fps
            )

            logger.warning("Using pre-inferred action captions. The pre-inferred action captions "
                           "should represent intervals of 1 second for each second of the video (180 captions for EgoSchema).")

        elif self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa":

            # infer action captions from the video clip using LaViLa
            sample_rate = self.get_action_captions_from_video_clip_config["resample_rate"]

            # get the full video tensor data
            resampled_video_clip = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
            video_clip_data = resampled_video_clip.data
            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using LaViLa
            action_captions = infer_transcript_from_video_clip_using_action_captions(
                video_clip_data,
                start_frame=start_frame,
                fps=sample_rate,
                original_fps=video_clip.original_fps,
                interval_in_seconds=self.get_action_captions_from_video_clip_config["interval_in_seconds"],
                temperature=self.get_action_captions_from_video_clip_config["temperature"],
                top_p=self.get_action_captions_from_video_clip_config["top_p"],
                max_text_length=self.get_action_captions_from_video_clip_config["max_new_tokens"],
                num_return_sequences=self.get_action_captions_from_video_clip_config["num_return_sequences"],
                early_stopping=self.get_action_captions_from_video_clip_config["early_stopping"],
                num_segments=self.get_action_captions_from_video_clip_config["num_segments"],
                cuda=True,
                modelzoo_dir_path=self.get_action_captions_from_video_clip_config["modelzoo_dir_path"],
                checkpoint_download_url=self.get_action_captions_from_video_clip_config["checkpoint_download_url"],
                checkpoint_file=self.get_action_captions_from_video_clip_config["checkpoint"]
            )

            # reduce to the very first action caption of each interval for now
            reduced_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(reduced_action_captions.keys(), reduced_action_captions))

        elif self.get_action_captions_from_video_clip_config["model_name"] == "CogAgent":

            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using CogAgent
            action_captions = infer_transcript_from_video_clip_using_frame_captions(
                video_clip.data,
                start_frame=start_frame,
                original_fps=video_clip.original_fps,
                frame_prompt=self.get_action_captions_from_video_clip_config["frame_prompt"],
                model_id=self.get_action_captions_from_video_clip_config["model_id"],
                tokenizer_id=self.get_action_captions_from_video_clip_config["tokenizer_id"],
                device=self.get_action_captions_from_video_clip_config["device"],
                precision=self.get_action_captions_from_video_clip_config["precision"],
                quantization=self.get_action_captions_from_video_clip_config["quantization"],
                temperature=self.get_action_captions_from_video_clip_config["temperature"],
                max_new_tokens=self.get_action_captions_from_video_clip_config["max_new_tokens"],
                do_sample=self.get_action_captions_from_video_clip_config["do_sample"]
            )

            # reduce to the very first action caption of each interval for now
            reduced_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(action_captions.keys(), reduced_action_captions))

        assert final_action_captions is not None, "Action captions should have been inferred."
        return list(final_action_captions.values())

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
            video=video_clip.data,
            text=text,
            config_dir=self.get_temporal_grounding_from_video_clip_and_text_config["config_dir"],
            checkpoint=self.get_temporal_grounding_from_video_clip_and_text_config["checkpoint"],
            clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config["clip_model_version"],
            output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config["output_feat_size"],
            half_precision=self.get_temporal_grounding_from_video_clip_and_text_config["half_precision"],
            jit=self.get_temporal_grounding_from_video_clip_and_text_config["jit"],
            resize_size=self.get_temporal_grounding_from_video_clip_and_text_config["resize"],
            gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config["gpu_id"]
        )

        foreground_indicators = text_temporal_grounding["foreground_indicators"]
        boundary_offsets = text_temporal_grounding["boundary_offsets"]
        saliency_scores = text_temporal_grounding["saliency_scores"].squeeze()

        # make a list of the foreground indicators
        foreground_indicators_list = [foreground_indicators[i].item() for i in range(foreground_indicators.shape[0])]
        logger.debug("Derived foreground indicators")

        # derive the best boundary offset indices
        k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        
        _, top_k_indices = torch.topk(boundary_offsets.flatten(), k=k)

        top_k_indices = top_k_indices.tolist()

        # initialize the relevance indicators with zeros
        relevance_indicators = [0 for _ in range(len(video_clip))]

        # iteratively update the relevance indicators for the top k intervals
        for top_i_index in top_k_indices:
            top_i_boundary_offset = boundary_offsets[top_i_index].tolist()
            logger.debug(f"Top {top_i_index} boundary offset: {top_i_boundary_offset}.")
            
            # optimistic flooring of start index
            start_index = max(0, top_i_index + math.floor(top_i_boundary_offset[0] * len(video_clip)))
            logger.debug(f"Start index: {start_index}.")

            # optimistic ceiling of end index
            end_index = min(start_index + math.ceil(top_i_boundary_offset[1] * len(video_clip)), len(video_clip) - 1)
            logger.debug(f"End index: {end_index}.")

            # update the relevance indicators
            relevance_indicators = [1 if start_index <= i <= end_index else relevance_indicators[i] for i in range(len(video_clip))]
            logger.debug(f"Relevance indicators: {relevance_indicators}")
        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        salience_indicators = []

        num_saliency_scores = 1 if saliency_scores.dim() == 0 else saliency_scores.size(0)
        for i in range(num_saliency_scores):
            saliency_score = saliency_scores[i].item() if num_saliency_scores > 1 else saliency_scores.item()
            saliency_score = max(0.0, min(1.0, saliency_score))
            salience_indicators.append(saliency_score)
        logger.debug("Derived salience indicators")

        return {
            "foreground_indicators": foreground_indicators_list,
            "relevance_indicators": relevance_indicators,
            "salience_indicators": salience_indicators
        }