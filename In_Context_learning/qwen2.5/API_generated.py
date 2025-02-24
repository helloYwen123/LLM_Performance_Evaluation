# #######################################################
# #######################################################
# Based on Description to generate code
# Here is an example for coding based on description:
#              \n\n## example description: 
#              \n A class named Expand, which inherit Operation, it has inputs num_splits: int = 4 and member variables self.num_splits is initialized  with inputs
# its function is named _execute takes graph: Optional[Graph], api: Optional[API], target: Optional[Node] as inputs and return None.
# get source_video_clip from target.state.video_clip, generating indices and splits from torch.tensor(list(range(len(source_video_clip)))) and torch.chunk(indices, self.num_splits)
# initialize target_nodes as [] and iterate in splits , pick each element as split
# get the start and end index of the split and then trim the video data to the split with start_list_index and end_list_index + 1 
# and then trim the sampled indices to the split using source_video_clip.sampled_indices[start_list_index:end_list_index + 1]
# and then create a new video clip from the split in a class VideoClip , whose inputs include data=split_data,path=source_video_clip.path,original_fps=source_video_clip.original_fps,original_num_frames=source_video_clip.original_num_frames,sampled_fps=source_video_clip.sampled_fps,sampled_indices=split_sampled_indices
# create a new node with new state instance for the new node via calling Node class, its inputs includes video_clip,task,lexical_representation,spatial_node_state and temporal_node_state
# then append the gotten results new_node to target_nodes list
# next initialize the class Edge like: [Edge(source=source_node, target_node=target_node) for target_node in target_nodes for source_node in [target]]
# and add this new Edge to the graph using add_nodes and add_edges and logger info after this
# the second function is __str__ it will return self.num_splits
# \n\n## example code answer:
# \n class Expand(Operation):
#     def __init__(self, num_splits: int = 4):
#         super().__init__()
#         self.num_splits = num_splits

#     def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
#         source_video_clip = target.state.video_clip

#         indices = torch.tensor(list(range(len(source_video_clip))))
#         splits = torch.chunk(indices, self.num_splits)

#         target_nodes = []
#         for split in splits:
#             get the start and end index of the split
#             start_list_index = split[0].item()
#             end_list_index = split[-1].item()

#             trim the video data to the split
#             split_data = source_video_clip.data[start_list_index:end_list_index + 1]

#             trim the sampled indices to the split
#             split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

#             create a new video clip from the split
#             split_video_clip = VideoClip(
#                 data=split_data,
#                 path=source_video_clip.path,
#                 original_fps=source_video_clip.original_fps,
#                 original_num_frames=source_video_clip.original_num_frames,
#                 sampled_fps=source_video_clip.sampled_fps,
#                 sampled_indices=split_sampled_indices
#             )

#             create a new node with new state for the new node
#             new_node = Node(
#                 state=NodeState(
#                     video_clip=split_video_clip,
#                     task=target.state.task,
#                     lexical_representation=target.state.lexical_representation,
#                     spatial_node_state=SpatialNodeState(
#                         video_clip=split_video_clip,
#                         task=target.state.spatial_node_state.task,
#                         use_action_captions=target.state.spatial_node_state.use_action_captions,
#                         use_object_detections=target.state.spatial_node_state.use_object_detections,
#                         use_action_captions_summary=target.state.spatial_node_state.use_action_captions_summary,
#                         use_object_detections_summary=target.state.spatial_node_state.use_object_detections_summary,
#                         lexical_representation=target.state.spatial_node_state.lexical_representation
#                     ),
#                     temporal_node_state=TemporalNodeState(
#                         video_clip=split_video_clip,
#                         task=target.state.temporal_node_state.task,
#                         use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
#                         lexical_representation=target.state.temporal_node_state.lexical_representation,
#                         use_relevance=target.state.temporal_node_state.use_relevance,
#                         use_foreground=target.state.temporal_node_state.use_foreground,
#                         use_salience=target.state.temporal_node_state.use_salience
#                     )
#                 )
#             )

#             target_nodes.append(new_node)

#         apply the expansion to the graph
#         edges = [Edge(source=source_node, target_node=target_node)
#                  for target_node in target_nodes
#                  for source_node in [target]]

#         graph.add_nodes(target_nodes)
#         graph.add_edges(edges)

#         logger.info(f"Executed structure expansion operation: Expand")

#     def __str__(self):
#         return f"Expand(num_splits={self.num_splits})"
#     \n\n Now is your turn, please finish code based on description:
#     \n## Description: 
# #######################################################
# #######################################################

class API():
    def __init__(
        self,
        get_object_detections_from_video_clip_and_text_config: Dict[str, Any],
        get_action_captions_from_video_clip_config: Dict[str, Any],
        get_temporal_grounding_from_video_clip_and_text_config: Dict[str, Any],
        get_summary_from_noisy_perceptive_data_config: Dict[str, Any],
        get_completion_from_text_config: Dict[str, Any],
        get_unspecific_objects_from_video_clip_config: Dict[str, Any],
        get_specific_objects_from_video_clip_config: Dict[str, Any],
        random_seed: int,
        reset_seed_for_each_function: bool = True,
        load_models_only_when_needed: bool = False
    ) -> None:
        
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
        """
        Initializes language models (LMLMs) according to configuration settings provided during initialization.
        """
        available_llm_classes = {
            'HuggingFaceLLM': lambda x: HuggingFaceLLM(x),
            'OpenAILLM': lambda x: OpenAILLM(x)
        }
        llm_class_to_use = self.get_completion_from_text_config['llm_class']
        self.llm = available_llm_classes[llm_class_to_use](self.get_completion_from_text_config)
        if not self.load_models_only_when_needed:
            self.llm.build_model()
            
    @property
    def llm(self) -> Union[HuggingFaceLLM, OpenAILLM]:
        """Returns current LLM model."""
        return self._llm
    
    @llm.setter
    def llm(self, val: Union[HuggingFaceLLM, OpenAILLM]):
        """Sets/updates currently used LLM model"""
        self._llm = val
            
    def reset_seed(self) -> None:
        """
        Resets internal states such that they don't carry over across different calls to `API`.
        """
        try:
            del os.environ['CUDA_LAUNCH_BLOCKING']  
        except KeyError:
            pass
        torch.cuda.empty_cache()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print('Random seed is reset.')
        
    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip) -> object:
        """
        Returns bounding boxes inferred from both video content & textual information.
        """
        self.reset_seed()
        return infer_bounding_boxes_from_video(video_clip)
    
    def get_action_captions_from_video_clip(self, video_clip: VideoClip) -> List[str]:
        """
        Generates action captions from video clips.
        """
        self.reset_seed()
        final_action_captions = None
        if self.get_action_captions_from_video_clip_config['model_name'] != 'LaViLa' \
           and self.get_action_captions_from_video_clip_config['model_name'] != 'CogAgent':
            raise ValueError(f'Model name must either LaViLa or CogAgent but got '
                             f'{self.get_action_captions_from_video_clip_config["model_name"]}.')

        if self.get_action_captions_from_video_clip_config['pre_inferred_action_captions_path']:
            pre_inferred_action_captions_dict = read_json_file(
                self.get_action_captions_from_video_clip_config[
                    'pre_inferred_action_captions_path'])
            video_clip_data = get_clip_data_from_video_data(video_clip)
            start_frame = len(pre_inferred_action_captions_dict.keys())
            action_captions = [
                pre_inferred_action_captions_dict[i]['caption']
                for i in sorted(pre_inferred_action_captions_dict)]
            if self.get_action_captions_from_video_clip_config['model_name'] == 'LaViLa':
                resample_rate = self.get_action_captions_from_video_clip_config[
                    'resample_rate']
                action_captions = infer_transcript_from_video_clip_using_action_captions(
                    video_clip_data, start_frame=start_frame * resample_rate // frame_size,
                    end_frame=(start_frame - 1) * resample_rate // frame_size +
                              len(action_captions))
            elif self.get_action_captions_from_video_clip_config['model_name'] == 'CogAgent':
                action_captions = infer_transcript_from_video_clip_using_frame_captions(
                    video_clip_data, start_frame=start_frame//frame_size+1,
                    end_frame=len(action_captions)//frame_size+2)
                
        else:
            action_captions = None
                
        if action_captions is None:
            raise ValueError('Action Captions cannot be generated due to missing pre-inferred action captures')
        reduced_action_captions = {}
        prev_start_time = float('-inf')
        curr_idx = 0
        while curr_idx < len(action_captions):
            start_time = action_captions[curr_idx][0]
            duration = action_captions[curr_idx][3]
            if start_time > prev_start_time + duration:
                break
            prev_start_time += duration
            reduced_action_captions.update({curr_idx: {'caption':action_captions[curr_idx][5]}})
            curr_idx += 1
        assert(reduced_action_captions), 'Final Action Captions cannot be generated'
        return list(reduced_action_captions.values())

def infer_bounding_boxes_from_video(video_clip: VideoClip) -> object:
    """
    Placeholder method for inferring bounding boxes from videos.
    """
    # Implementation details would go here...
    return None

def read_json_file(file_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Reads JSON file at specified location.
    """
    with open(file_path, mode='r') as infile:
        return json.load(infile)

def get_clip_data_from_video_data(video_clip: VideoClip) -> np.ndarray:
    """
    Extracts relevant features from video data.
    """
    # Implementation specifics depend upon how video data is structured.
    return np.array([video_clip.frame_features])

def infer_temporal_grounding_score_from_video_and_text(video_clip: VideoClip, text: str) -> Dict[str, Any]:
    """
    Computes grounding scores related to specific intervals within the video.
    """
    return infer_temporal_grounding_score_from_video_and_text(video_clip, text)

# #######################################################
# #######################################################
# Completion given Half of the code
# \n\n Now i will give you the part of one class which i have finished, help me to complete it based on the above requirment and description and give me the completed code finally.
# Now i will give you the part of one class which i have finished, help me to complete it based on the requirment and description and give me the completed code finally.
#              \nHere is an example for coding based on description and partial Ground Truth code:
#              \n\n## example description: 
#              \n A class named Expand, which inherit Operation, it has inputs num_splits: int = 4 and member variables self.num_splits is initialized  with inputs
# its function is named _execute takes graph: Optional[Graph], api: Optional[API], target: Optional[Node] as inputs and return None.
# get source_video_clip from target.state.video_clip, generating indices and splits from torch.tensor(list(range(len(source_video_clip)))) and torch.chunk(indices, self.num_splits)
# initialize target_nodes as [] and iterate in splits , pick each element as split
# get the start and end index of the split and then trim the video data to the split with start_list_index and end_list_index + 1 
# and then trim the sampled indices to the split using source_video_clip.sampled_indices[start_list_index:end_list_index + 1]
# and then create a new video clip from the split in a class VideoClip , whose inputs include data=split_data,path=source_video_clip.path,original_fps=source_video_clip.original_fps,original_num_frames=source_video_clip.original_num_frames,sampled_fps=source_video_clip.sampled_fps,sampled_indices=split_sampled_indices
# create a new node with new state instance for the new node via calling Node class, its inputs includes video_clip,task,lexical_representation,spatial_node_state and temporal_node_state
# then append the gotten results new_node to target_nodes list
# next initialize the class Edge like: [Edge(source=source_node, target_node=target_node) for target_node in target_nodes for source_node in [target]]
# and add this new Edge to the graph using add_nodes and add_edges and logger info after this
# the second function is __str__ it will return self.num_splits
# \n\n## example for partial Ground Truth code:
# \n class Expand(Operation):
#     def __init__(self, num_splits: int = 4):
#         super().__init__()
#         self.num_splits = num_splits

#     def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
#         source_video_clip = target.state.video_clip

#         indices = torch.tensor(list(range(len(source_video_clip))))
#         splits = torch.chunk(indices, self.num_splits)

#         target_nodes = []
#         for split in splits:
#             # get the start and end index of the split
#             start_list_index = split[0].item()
#             end_list_index = split[-1].item()

#             # trim the video data to the split
#             split_data = source_video_clip.data[start_list_index:end_list_index + 1]

#             # trim the sampled indices to the split
#             split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

#             # create a new video clip from the split
#             split_video_clip = VideoClip(
#                 data=split_data,
#                 path=source_video_clip.path,
#                 original_fps=source_video_clip.original_fps,
#                 original_num_frames=source_video_clip.original_num_frames,
#                 sampled_fps=source_video_clip.sampled_fps,
#                 sampled_indices=split_sampled_indices
#             )

#             # create a new node with new state for the new node


#         graph.add_nodes(target_nodes)
#         graph.add_edges(edges)

#         logger.info(f"Executed structure expansion operation: Expand")

#     def __str__(self):
#         return f"Expand(num_splits={{self.num_splits}})"
# \n\n## example code answer:
# \n class Expand(Operation):
#     def __init__(self, num_splits: int = 4):
#         super().__init__()
#         self.num_splits = num_splits

#     def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
#         source_video_clip = target.state.video_clip

#         indices = torch.tensor(list(range(len(source_video_clip))))
#         splits = torch.chunk(indices, self.num_splits)

#         target_nodes = []
#         for split in splits:
#             # get the start and end index of the split
#             start_list_index = split[0].item()
#             end_list_index = split[-1].item()

#             # trim the video data to the split
#             split_data = source_video_clip.data[start_list_index:end_list_index + 1]

#             # trim the sampled indices to the split
#             split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

#             # create a new video clip from the split
#             split_video_clip = VideoClip(
#                 data=split_data,
#                 path=source_video_clip.path,
#                 original_fps=source_video_clip.original_fps,
#                 original_num_frames=source_video_clip.original_num_frames,
#                 sampled_fps=source_video_clip.sampled_fps,
#                 sampled_indices=split_sampled_indices
#             )

#             # create a new node with new state for the new node
#             new_node = Node(
#                 state=NodeState(
#                     video_clip=split_video_clip,
#                     task=target.state.task,
#                     lexical_representation=target.state.lexical_representation,
#                     spatial_node_state=SpatialNodeState(
#                         video_clip=split_video_clip,
#                         task=target.state.spatial_node_state.task,
#                         use_action_captions=target.state.spatial_node_state.use_action_captions,
#                         use_object_detections=target.state.spatial_node_state.use_object_detections,
#                         use_action_captions_summary=target.state.spatial_node_state.use_action_captions_summary,
#                         use_object_detections_summary=target.state.spatial_node_state.use_object_detections_summary,
#                         lexical_representation=target.state.spatial_node_state.lexical_representation
#                     ),
#                     temporal_node_state=TemporalNodeState(
#                         video_clip=split_video_clip,
#                         task=target.state.temporal_node_state.task,
#                         use_temporal_grounding_summary=target.state.temporal_node_state.use_temporal_grounding_summary,
#                         lexical_representation=target.state.temporal_node_state.lexical_representation,
#                         use_relevance=target.state.temporal_node_state.use_relevance,
#                         use_foreground=target.state.temporal_node_state.use_foreground,
#                         use_salience=target.state.temporal_node_state.use_salience
#                     )
#                 )
#             )

#             target_nodes.append(new_node)

#         # apply the expansion to the graph
#         edges = [Edge(source=source_node, target_node=target_node)
#                  for target_node in target_nodes
#                  for source_node in [target]]

#         graph.add_nodes(target_nodes)
#         graph.add_edges(edges)

#         logger.info(f"Executed structure expansion operation: Expand")

#     def __str__(self):
#         return f"Expand(num_splits={self.num_splits})"
#     \n\n Now is your turn, please finish code based on description:
#     \n## Description: 
    
#     \n\n ## the partial Ground Truth code:
    
    
#     \n\n now please write complete code which is needed based on above mentioned. Remember you need to output complete code which fully meets the requirement and description without any conversational text
# #######################################################
# #######################################################
class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: dict[str, any],
            get_action_captions_from_video_clip_config: dict[str, any],
            get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
            get_summary_from_noisy_perceptive_data_config: dict[str, any],
            get_unspecpecific_objects_from_video_clip_config: dict[str, any],
            get_completion_from_text_config: dict[str, any],
            get_specific_objects_from_video_clip_config: dict[str, any],
            random_seed: int,
            reset_seed_for_each_function: bool = True,
            load_models_only_when_needed: bool = False
    ):

        self._initialize_llm()

    def _initialize_llm(self):

        available_llm_classes = {
            "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        self.llm = available_llm_classes[self.get_completion_from_text_config["llm_class"]]
        (
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

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        return infer_bounding_boxes_from_video(
            video_tensor=video_clip.data,
            obj=text,
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
            all_action_captions = read_json_file(file_path=self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"])
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
            relevance_indicators = [
                1 if start_index <= i <= end_index else relevance_indicators[i] for i in range(len(video_clip))
            ]
            logger.debug(f"Relevance indicators: {relevance_indicators}.")
        logger.debug(f"Derived relevance indicators: {relevance_indicators}")
# #######################################################
# #######################################################
# Filling blanks in the code
# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description. Give me the completed code finally.

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
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        llm = available_llm_classes[self.get_completion_from_text_config["llm_class"]](self.get_completion_from_text_config)

        # if the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            llm.build_model()

    def reset_seed(self):
        # clear caches
        torch.cuda.empty_cache()
        importlib.invalidate_caches()
        logger.info("Cleared CUDA cache and invalidated imports.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        return (
            video_tensor=video_clip.data,
            object_detection=text,
            config_file=self.get_object_detections_from_video_clip_and_text_config["config_file"],
            checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
            box_threshold=self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
            text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
            cuda=self.get_object_detections_from_video_clip_and_text_config["cuda"]
        )

    def get_action_captions_from_video_clip(self, video_clip: VideoClip):
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
                fps=video_clip.original_fps
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
                checkpoint_file=self.get_action_captions_from_video_clip_config["checkpoint"]
            )

            # reduce to the very first action caption of each interval for now
            reduced_action_captions = [action_captions[0][key] for key in action_captions.keys()]

            final_action_captions = dict(zip(reduced_action_captions.keys(), reduced_action_captions.values()))

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
            reduced_action_captions = [action_captions[key][0] for key in action_captions.keys()]

            final_action_captions = dict(zip(reduced_action_captions.keys(), reduced_action_captions))

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
            
            start_index = max(0, round(top_i_boundary_offset[0] * len(video_clip)))
            end_index = min(round((top_i_boundary_offset[1] + 1) * len(video_clip)), len(video_clip)-1)
            logger.debug(f"Start index: {start_index}, End index: {end_index}")
            
            relevance_indicators = [
                1 if s >= start_index and e <= end_index else relevance_indicators[s]  
                for s,e in zip(range(len(video_clip)),range(len(video_clip)+len(video_clip)))[:2*len(video_clip)]
            ]
            logger.debug(f"Relevance indicators updated: {relevance_indicators}")

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