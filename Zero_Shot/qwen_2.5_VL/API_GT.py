############# GT ##############
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
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        self.llm = available_llm_classes[
            self.get_completion_from_text_config["llm_class"]](
            self.get_completion_from_text_config
        )

        # if the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            self.llm.build_model()

    def reset_seed(self):
        # clear caches
        torch.cuda.empty_cache()
        importlib.invalidate_caches()
        logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        return infer_bounding_boxes_from_video(
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
            all_action_captions = read_json_file(
                file_path=self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"])
            action_captions = all_action_captions[video_clip.id]

            final_action_captions = get_clip_data_from_video_data(
                video_data=action_captions,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )

            logger.warning("Using pre-inferred action captions. The pre-inferred action captions "
                           "should represent intervals of 1 second for each second of the video (180 captions for"
                           "EgoSchema).")

        elif self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa":

            # infer action captions from the video clip using LaViLa
            sample_rate = self.get_action_captions_from_video_clip_config["resample_rate"]

            # get the full video tensor data
            resampled_video_clip = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
            video_clip_data = resampled_video_clip.data
            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using LaViLa
            action_captions = infer_transcript_from_video_clip_using_action_captions(
                video_clip=video_clip_data,
                start_frame=start_frame,
                fps=sample_rate,
                original_fps=video_clip.original_fps,
                interval_in_seconds=self.get_action_captions_from_video_clip_config["interval_in_seconds"],
                temperature=self.get_action_captions_from_video_clip_config["temperature"],
                top_p=self.get_action_captions_from_video_clip_config["top_p"],
                max_text_length=self.get_action_captions_from_video_clip_config["max_new_tokens"],
                num_return_sequences=self.get_action_captions_from_video_clip_config["num_return_sequences"],
                early_stopping=self.get_action_captions_from_video_clip_config["early_stopping"],
                num_seg=self.get_action_captions_from_video_clip_config["num_seg"],
                cuda=True,
                modelzoo_dir_path=self.get_action_captions_from_video_clip_config["modelzoo_dir_path"],
                checkpoint_download_url=self.get_action_captions_from_video_clip_config["checkpoint_download_url"],
                checkpoint_file=self.get_action_captions_from_video_clip_config["checkpoint"]
            )

            # reduce to the very first action caption of each interval for now
            # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
            final_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(action_captions.keys(), final_action_captions))

        elif self.get_action_captions_from_video_clip_config["model_name"] == "CogAgent":

            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using CogAgent
            action_captions = infer_transcript_from_video_clip_using_frame_captions(
                video_clip=video_clip.data,
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
            # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
            final_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(action_captions.keys(), final_action_captions))

        assert final_action_captions is not None, "Action captions should have been inferred."
        return list(final_action_captions.values())

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
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
        saliency_scores = text_temporal_grounding["saliency_scores"].squeeze()

        # make a list of the foreground indicators
        foreground_indicators_list = [foreground_indicators[i].item() for i in range(foreground_indicators.size(0))]
        logger.debug("Derived foreground indicators")

        # derive the best boundary offset indices
        k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        logger.debug(f"Deriving top {k} boundary offsets.")
        logger.debug(f"Boundary offsets: {boundary_offsets}.")
        logger.debug(f"Foreground indicators: {foreground_indicators}.")
        logger.debug(f"Flattened foreground indicators: {foreground_indicators.flatten()}.")
        _, top_k_indices = torch.topk(foreground_indicators.flatten(), k=k)
        logger.debug(f"Top {k} indices: {top_k_indices}.")
        top_k_indices = top_k_indices.tolist()
        logger.debug(f"Top {k} indices (converted to list): {top_k_indices}.")

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
            end_index = min(top_i_index + math.ceil(top_i_boundary_offset[1] * len(video_clip)), len(video_clip) - 1)
            logger.debug(f"End index: {end_index}.")

            # update the relevance indicators
            # i.e., set all relevance indicators between start and end index to 1
            relevance_indicators = [1 if start_index <= i <= end_index else relevance_indicators[i] for i in
                                    range(len(video_clip))]
            logger.debug(f"Relevance indicators: {relevance_indicators}.")
        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        # deprecated derivation of top 1 boundary offset only
        # top_1_index = torch.argmax(foreground_indicators)
        # top_1_boundary_offset = boundary_offsets[top_1_index].tolist()
        # optimistic flooring of start index
        # top_1_start_index = max(0, top_1_index + math.floor(top_1_boundary_offset[0] * len(video_clip)))
        # optimistic ceiling of end index
        # top_1_end_index = min(top_1_index + math.ceil(top_1_boundary_offset[1] * len(video_clip)), len(video_clip) - 1)
        # set the relevance indicators to 1 for the interval, 0 for rest
        # relevance_indicators = [1 if top_1_start_index <= i <= top_1_end_index else 0 for i in range(len(video_clip))]

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