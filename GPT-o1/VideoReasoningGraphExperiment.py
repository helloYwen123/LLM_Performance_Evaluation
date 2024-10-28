# GPT-o1-preview

# here, you are an very experienced Programmer, who is very good at programming under others' instruction
#  and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# help me to write a class called VideoReasoningExperiment, 
# it has father-class named VideoReasoningExperiment,  this class take config:dict as its initialization's input
# class can use cuda and gpu for following computation if aviable
# class needs to initialize many members variables in __init__: they are following
# videos_path;tasks_path;answers_path;predictions_path
# conclusions_path 
# decisions_path 
# accuracy_path 
# tates_path 
# resume_path 
# iterate_through_videos
# sample_rate 
# start_video 
# end_video 
# subset_indices 
# max_depth 
# random_seed
# reset_seed_for_each_function
# reset_seed_for_each_video
# spatial_node_state_config
# temporal_node_state_config
# lexical_representation
# and the initialization needs also some several member functions: they are following
# _create_experiment_paths_and_files(config)
# _initialize_operations(config.get("operations"))
# _initialize_api(config.get("api"))
# and like :data = load_data(answer_path=, task_path=, nomalize=)
# execute the :self.predictions = read_json_file(file_path=self.predictions_path) if experiment is not a new one.
# class has functionn named _create_experiment_paths_and_files, it takes config as input and returns nothing
# it creates experiments paths and files. if there is no resume path for an old experiment. it create paths variable and save it from path list:predictions_path,conclusion_path and decisions_path
# and then use write_json_file function to create json file in given paths. then logger info after successful create.
# if there exists resume paths, the files in resume_paths will be copied into created file in experiment_paths.
# the existed resume paths includes folder named "states" and json file.
# next class has funciton named initialize_operations() taking operation_config as input parameters.
# function has a dictionary , its keys is string ; value are classes for different operations, including Expand;DPCKNNExpand;Wait;Merge;DeriveNodeStates;DeriveRootNodeState;Ranking;NoRatings;SummaryConfidenceRating;AnswerabilityRating;NoDecision;PairwiseHammingDistanceDecision;OptionsCandidateBasedOnConcatenatedLexicalStates;HammingSimilarityOfOptionsRankingsPerNodeConclusion;OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates;MajorityVoteOfRankingsConclusion;NoConclusion;and last is IterativeMergeConclusion
# what's more, every operation in dictionary will be called via respective keys and assigned to class memeber variables above mentioned: split_operation wait_operation, etc. their names has part of "operation"
# self.decision and self.conclusion are also two of operations. 
# These self member variables' assignment are all from available_opeartions dictionary via input operation_config and .get
# such as config.get("").get("") ,and finally using **operations_config.get("").get("params") 
# when assigning self.decision, inputs includes split_operation,merge_operation and wait_operation
# next is function named _initialize_api and its input is called api_config
# firstly get_completion_from_text_config is assigned from api_config.get
# llm_class if from get_completion_from_text_config.get("llm_class")
# if “openai" in llm_class, then  self.random_seed will be setted to one of elements in get_completion_from_text_config
# finally, use imported module name API to set self.api. API takes following as inputs
# including load_models_only_when_needed;get_object_detections_from_video_clip_and_text_config;get_action_captions_from_video_clip_config;get_summary_from_noisy_perceptive_data_config;get_completion_from_text_config;get_unspecific_objects_from_video_clip_config;get_specific_objects_from_video_clip_config;random_seed;reset_seed_for_each_function
# next they are three function:_train;_eval;_test.
# among them only _eval has conrete implementation. when _train and _test are run, they will report error explaning currently these two function are invalid.
# the function firstly call time.time() to record timepoint at moment.
# initialize num_samples_correct, num_samples_total and visited_video_ids
# then use enumerate(self.data) to do the loop and iteratively get current video_count and item. item has three members:video_id;question;options
# then use the item's members to assgin and video_id, question, and options
# remember here to skip videos if video_count < self.start_video, videos which have already been processed and self.iterate_through_videos is True or its video_count is not in the specified subset indices
# save current video_id in visited_video_ids. and clear all cache like cuda's or any else.
# if reset_seed_for_each_video is true and video_count is start of the video,set the random seed (video-wise, so we can reproduce results for a certain video)
# moreover, the seed will be re-set in the api for each "neural module" call, lik cpu, gpu, python library and numpy
# then logger info the current loop's id of reasoned video and given question, options
# from videos_path create path for each video in that folder.
# carefully do when video_path is none. 
# and then use create_video_data_from_video_path to get variables: video, metadata and debug "original sampled indices: {metadata['sample_indices']}"
# transform video to video_clip via Class VideoClip's function from_meatadata, taking the above video and metada as inputs
# info then the current video_path, self.sample_rate and len of video_clip (unit is frame)
# create the states save path for the current video and initialize the task via Task class with input gotten quetion and options.
# then initialize another class named NodeState and outputs as root_state
# its needed inputs are: video_clip;task;lexical_representation; and two other initialized classes, which are SpatialNodeState and TemporalNodeState
# SpatialNodeState's initializing inputs are video_clip, task, use_action_captionstask, use_object_detections, use_action_captions_summary, use_object_detections_summary, lexical_representation
# TemporalNodeState‘s ones are video_clip;task;lexical_representation;use_foreground;use_relevance;use_salience;use_temporal_grounding_summary
# respectively they get mostly needed value from self.spatial_node_state_config.get() and self.temporal_node_state_config.get()
# then use aboved gotten variable root_state to initialize Node class in the way as(state=root_state)
# and then initialize another class named Graph with root_node in the same way
#  initialize the video reasoning controller class 
# VideoReasoningController needs following inputs: video_reasoning_graph; task; api; states_save_path; expand_operation;merge_operation;derive_node_state_operation;derive_universal_state_operation;rating;decision;conclusion;max_depth;predictions_save_path;conclusions_save_path;decisions_save_path
# execute the video reasoning through single_choice_reasoning function and assign result to choice variable
# record the correct answers and total number of iteration. and then calculate the accuracy of reasoning results and info it in terminal.
# measure the execution time per iteration / video and also report it in info
# get all_predictions from read_json_file in the path of self.predictions_path. Then iterate all_predictions to get video_predicitons and video_id
# iterate the video_predicitions.items() to get question and question_prediction. and iteratively check if item["video_id"] and item["question"] are right in this turn
# if item in self.data is corresponding to the extracted info from all_predictions in this iteration
# then take the first element in item as corresponding_data_entry and assign corresponding_data_entry["answer"] to ground_truth
# compare the ground truth with the prediction for each iteration if prediction == ground_truth then num_correct_per_iteration will +1 ,calculate the number of correct answer per iteration saving as num_correct_per_iteration[index], index is iteration -1
# so actually question_prediction.items() has iteration and prediction two objects
# if the list num_coorect_per_iteration has element 0, assgin it with element of last index
# calculate the accuracy per iteration saving as accuracy_per_iteration and info :umber of correct predictions for iteration; total number of predictions for this iteration; and Accuracy for iteration
# save the accuracies using write_json_file which takes data: accuracy_per_iteration and file_path:accuracy_path as input
# measure the total execution time using time.time() and info the total executation time of whole experiment
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
import os
import time
import json
import logging
import shutil
import random
import numpy as np
import torch

# Assume these modules are implemented elsewhere
from api_module import API
from video_clip_module import VideoClip
from node_state_module import NodeState, SpatialNodeState, TemporalNodeState
from node_module import Node
from graph_module import Graph
from task_module import Task
from video_reasoning_controller_module import VideoReasoningController
from operations_module import (
    Expand, DPCKNNExpand, Wait, Merge, DeriveNodeStates, DeriveRootNodeState, Ranking, NoRatings,
    SummaryConfidenceRating, AnswerabilityRating, NoDecision, PairwiseHammingDistanceDecision,
    OptionsCandidateBasedOnConcatenatedLexicalStates, HammingSimilarityOfOptionsRankingsPerNodeConclusion,
    OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates, MajorityVoteOfRankingsConclusion,
    NoConclusion, IterativeMergeConclusion
)

logger = logging.getLogger(__name__)

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data(answer_path, task_path, normalize):
    # Implement logic to load data from answer_path and task_path
    # Return a list of data items
    data_list = []  # Replace with actual data loading logic
    return data_list

def create_video_data_from_video_path(video_path):
    # Implement logic to load video and metadata
    video = None  # Load video
    metadata = {'sample_indices': []}  # Load metadata
    return video, metadata

class VideoReasoningExperiment:
    def __init__(self, config: dict):
        # Initialize member variables from config
        self.videos_path = config.get('videos_path')
        self.tasks_path = config.get('tasks_path')
        self.answers_path = config.get('answers_path')
        self.predictions_path = config.get('predictions_path')
        self.conclusions_path = config.get('conclusions_path')
        self.decisions_path = config.get('decisions_path')
        self.accuracy_path = config.get('accuracy_path')
        self.states_path = config.get('states_path')
        self.resume_path = config.get('resume_path')
        self.iterate_through_videos = config.get('iterate_through_videos', False)
        self.sample_rate = config.get('sample_rate', 1)
        self.start_video = config.get('start_video', 0)
        self.end_video = config.get('end_video', None)
        self.subset_indices = config.get('subset_indices', [])
        self.max_depth = config.get('max_depth', 10)
        self.random_seed = config.get('random_seed', 42)
        self.reset_seed_for_each_function = config.get('reset_seed_for_each_function', False)
        self.reset_seed_for_each_video = config.get('reset_seed_for_each_video', False)
        self.spatial_node_state_config = config.get('spatial_node_state_config', {})
        self.temporal_node_state_config = config.get('temporal_node_state_config', {})
        self.lexical_representation = config.get('lexical_representation', '')

        # Call initialization methods
        self._create_experiment_paths_and_files(config)
        self._initialize_operations(config.get("operations", {}))
        self._initialize_api(config.get("api", {}))

        # Load data
        self.data = load_data(
            answer_path=self.answers_path,
            task_path=self.tasks_path,
            normalize=True
        )

        # If experiment is not new, read existing predictions
        if not self._is_new_experiment():
            self.predictions = read_json_file(file_path=self.predictions_path)
        else:
            self.predictions = {}

    def _is_new_experiment(self):
        # Check if predictions file exists to determine if the experiment is new
        return not os.path.exists(self.predictions_path)

    def _create_experiment_paths_and_files(self, config):
        # Create experiment paths and files
        if not self.resume_path:
            # Create the directories and files
            paths = [self.predictions_path, self.conclusions_path, self.decisions_path]
            for path in paths:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Create an empty JSON file
                write_json_file({}, path)
            logger.info("Successfully created experiment paths and files.")
        else:
            # Resume from existing paths
            # Copy files from resume_path to the experiment paths
            shutil.copytree(os.path.join(self.resume_path, 'states'), self.states_path, dirs_exist_ok=True)
            json_files = ['predictions.json', 'conclusions.json', 'decisions.json']
            for json_file in json_files:
                src_file = os.path.join(self.resume_path, json_file)
                dst_file = getattr(self, json_file.replace('.json', '_path'))
                shutil.copyfile(src_file, dst_file)
            logger.info("Successfully resumed experiment from resume path.")

    def _initialize_operations(self, operation_config):
        available_operations = {
            'Expand': Expand,
            'DPCKNNExpand': DPCKNNExpand,
            'Wait': Wait,
            'Merge': Merge,
            'DeriveNodeStates': DeriveNodeStates,
            'DeriveRootNodeState': DeriveRootNodeState,
            'Ranking': Ranking,
            'NoRatings': NoRatings,
            'SummaryConfidenceRating': SummaryConfidenceRating,
            'AnswerabilityRating': AnswerabilityRating,
            'NoDecision': NoDecision,
            'PairwiseHammingDistanceDecision': PairwiseHammingDistanceDecision,
            'OptionsCandidateBasedOnConcatenatedLexicalStates': OptionsCandidateBasedOnConcatenatedLexicalStates,
            'HammingSimilarityOfOptionsRankingsPerNodeConclusion': HammingSimilarityOfOptionsRankingsPerNodeConclusion,
            'OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates': OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates,
            'MajorityVoteOfRankingsConclusion': MajorityVoteOfRankingsConclusion,
            'NoConclusion': NoConclusion,
            'IterativeMergeConclusion': IterativeMergeConclusion
        }

        # Assign operations to self member variables
        self.expand_operation = available_operations.get(
            operation_config.get('expand', {}).get('class'))(
            **operation_config.get('expand', {}).get('params', {})
        )
        self.merge_operation = available_operations.get(
            operation_config.get('merge', {}).get('class'))(
            **operation_config.get('merge', {}).get('params', {})
        )
        self.wait_operation = available_operations.get(
            operation_config.get('wait', {}).get('class'))(
            **operation_config.get('wait', {}).get('params', {})
        )
        self.derive_node_state_operation = available_operations.get(
            operation_config.get('derive_node_state', {}).get('class'))(
            **operation_config.get('derive_node_state', {}).get('params', {})
        )
        self.derive_universal_state_operation = available_operations.get(
            operation_config.get('derive_universal_state', {}).get('class'))(
            **operation_config.get('derive_universal_state', {}).get('params', {})
        )
        self.rating = available_operations.get(
            operation_config.get('rating', {}).get('class'))(
            **operation_config.get('rating', {}).get('params', {})
        )
        self.decision = available_operations.get(
            operation_config.get('decision', {}).get('class'))(
            expand_operation=self.expand_operation,
            merge_operation=self.merge_operation,
            wait_operation=self.wait_operation,
            **operation_config.get('decision', {}).get('params', {})
        )
        self.conclusion = available_operations.get(
            operation_config.get('conclusion', {}).get('class'))(
            **operation_config.get('conclusion', {}).get('params', {})
        )

    def _initialize_api(self, api_config):
        get_completion_from_text_config = api_config.get('get_completion_from_text_config', {})
        llm_class = get_completion_from_text_config.get('llm_class', '')
        if 'openai' in llm_class.lower():
            self.random_seed = get_completion_from_text_config.get('random_seed', self.random_seed)
        self.api = API(
            load_models_only_when_needed=api_config.get('load_models_only_when_needed', True),
            get_object_detections_from_video_clip_and_text_config=api_config.get('get_object_detections_from_video_clip_and_text_config', {}),
            get_action_captions_from_video_clip_config=api_config.get('get_action_captions_from_video_clip_config', {}),
            get_summary_from_noisy_perceptive_data_config=api_config.get('get_summary_from_noisy_perceptive_data_config', {}),
            get_completion_from_text_config=get_completion_from_text_config,
            get_unspecific_objects_from_video_clip_config=api_config.get('get_unspecific_objects_from_video_clip_config', {}),
            get_specific_objects_from_video_clip_config=api_config.get('get_specific_objects_from_video_clip_config', {}),
            random_seed=self.random_seed,
            reset_seed_for_each_function=self.reset_seed_for_each_function
        )

    def _train(self):
        raise NotImplementedError("Training function is currently invalid.")

    def _test(self):
        raise NotImplementedError("Testing function is currently invalid.")

    def _eval(self):
        start_time = time.time()
        num_samples_correct = 0
        num_samples_total = 0
        visited_video_ids = set()
        for video_count, item in enumerate(self.data):
            video_id = item['video_id']
            question = item['question']
            options = item['options']

            # Skip videos if necessary
            if video_count < self.start_video:
                continue
            if self.iterate_through_videos and video_id in visited_video_ids:
                continue
            if self.subset_indices and video_count not in self.subset_indices:
                continue

            visited_video_ids.add(video_id)
            # Clear cache (e.g., cuda)
            torch.cuda.empty_cache()
            # Reset seed if needed
            if self.reset_seed_for_each_video:
                seed = self.random_seed + video_count
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            logger.info(f"Processing video {video_id}: {question}, options: {options}")

            # Get video path
            video_path = os.path.join(self.videos_path, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video file {video_path} does not exist.")
                continue

            # Create video data from video path
            video, metadata = create_video_data_from_video_path(video_path)
            logger.debug(f"Original sampled indices: {metadata['sample_indices']}")

            # Transform video to video_clip
            video_clip = VideoClip.from_metadata(video, metadata)
            logger.info(f"Video path: {video_path}, sample rate: {self.sample_rate}, video clip length: {len(video_clip)} frames")

            # Create states save path for current video
            states_save_path = os.path.join(self.states_path, f"{video_id}")
            os.makedirs(states_save_path, exist_ok=True)

            # Initialize the task
            task = Task(question=question, options=options)

            # Initialize NodeState and other classes
            spatial_node_state = SpatialNodeState(
                video_clip=video_clip,
                task=task,
                use_action_captions=self.spatial_node_state_config.get('use_action_captions', False),
                use_object_detections=self.spatial_node_state_config.get('use_object_detections', False),
                use_action_captions_summary=self.spatial_node_state_config.get('use_action_captions_summary', False),
                use_object_detections_summary=self.spatial_node_state_config.get('use_object_detections_summary', False),
                lexical_representation=self.lexical_representation
            )
            temporal_node_state = TemporalNodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                use_foreground=self.temporal_node_state_config.get('use_foreground', False),
                use_relevance=self.temporal_node_state_config.get('use_relevance', False),
                use_salience=self.temporal_node_state_config.get('use_salience', False),
                use_temporal_grounding_summary=self.temporal_node_state_config.get('use_temporal_grounding_summary', False)
            )
            root_state = NodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                spatial_node_state=spatial_node_state,
                temporal_node_state=temporal_node_state
            )

            # Initialize Node and Graph
            root_node = Node(state=root_state)
            video_reasoning_graph = Graph(root_node=root_node)

            # Initialize the video reasoning controller
            controller = VideoReasoningController(
                video_reasoning_graph=video_reasoning_graph,
                task=task,
                api=self.api,
                states_save_path=states_save_path,
                expand_operation=self.expand_operation,
                merge_operation=self.merge_operation,
                derive_node_state_operation=self.derive_node_state_operation,
                derive_universal_state_operation=self.derive_universal_state_operation,
                rating=self.rating,
                decision=self.decision,
                conclusion=self.conclusion,
                max_depth=self.max_depth,
                predictions_save_path=self.predictions_path,
                conclusions_save_path=self.conclusions_path,
                decisions_save_path=self.decisions_path
            )

            # Execute the video reasoning
            choice = controller.single_choice_reasoning()

            # Record the correct answers and total number of iteration
            num_samples_total += 1
            if choice == item['answer']:
                num_samples_correct += 1

            # Calculate and log the accuracy
            accuracy = num_samples_correct / num_samples_total
            logger.info(f"Current accuracy: {accuracy:.2f}")

        # Measure execution time per iteration / video
        end_time = time.time()
        total_time = end_time - start_time
        if num_samples_total > 0:
            avg_time_per_video = total_time / num_samples_total
        else:
            avg_time_per_video = 0
        logger.info(f"Average time per video: {avg_time_per_video:.2f} seconds")

        # Evaluate overall predictions
        self._evaluate_predictions()

        # Measure the total execution time and log it
        logger.info(f"Total execution time: {total_time:.2f} seconds")

    def _evaluate_predictions(self):
        # Get all_predictions from read_json_file in the path of self.predictions_path.
        all_predictions = read_json_file(self.predictions_path)
        num_correct_per_iteration = []
        total_predictions_per_iteration = []
        accuracy_per_iteration = []

        for iteration, (video_id, video_predictions) in enumerate(all_predictions.items(), start=1):
            num_correct = 0
            total_predictions = 0
            for question, prediction in video_predictions.items():
                # Find the corresponding data entry in self.data
                corresponding_data_entry = next(
                    (item for item in self.data if item['video_id'] == video_id and item['question'] == question),
                    None
                )
                if not corresponding_data_entry:
                    continue
                ground_truth = corresponding_data_entry['answer']
                if prediction == ground_truth:
                    num_correct += 1
                total_predictions += 1

            if total_predictions == 0:
                # Assign the last known correct count if total predictions are zero
                if num_correct_per_iteration:
                    num_correct = num_correct_per_iteration[-1]
                    total_predictions = total_predictions_per_iteration[-1]
                else:
                    num_correct = 0
                    total_predictions = 0

            num_correct_per_iteration.append(num_correct)
            total_predictions_per_iteration.append(total_predictions)
            if total_predictions == 0:
                accuracy = 0
            else:
                accuracy = num_correct / total_predictions
            accuracy_per_iteration.append(accuracy)
            logger.info(f"Iteration {iteration}: {num_correct} correct predictions out of {total_predictions}, Accuracy: {accuracy:.2f}")

        # Save the accuracies
        write_json_file(accuracy_per_iteration, self.accuracy_path)

###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
###########################################################################################################################################################################################################

import os
import time
import json
import logging
import shutil
import random
import numpy as np
import torch
import importlib
from slugify import slugify  # Make sure to install python-slugify: pip install python-slugify

# Assume these modules are implemented elsewhere
from api_module import API
from video_clip_module import VideoClip
from node_state_module import NodeState, SpatialNodeState, TemporalNodeState
from node_module import Node
from graph_module import Graph
from task_module import Task
from video_reasoning_controller_module import VideoReasoningController
from operations_module import (
    Expand, DPCKNNExpand, Wait, Merge, DeriveNodeStates, DeriveRootNodeState, Ranking, NoRatings,
    SummaryConfidenceRating, AnswerabilityRating, NoDecision, PairwiseHammingDistanceDecision,
    OptionsCandidateBasedOnConcatenatedLexicalStates, HammingSimilarityOfOptionsRankingsPerNodeConclusion,
    OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates, MajorityVoteOfRankingsConclusion,
    NoConclusion, IterativeMergeConclusion
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_data(answer_path, task_path, normalize=True):
    # Implement your data loading logic here
    # For demonstration, returning an empty list
    return []

def create_video_data_from_video_path(video_path, sample_rate=1):
    # Implement your video loading and sampling logic here
    # For demonstration, returning dummy data
    video = None  # Replace with actual video loading
    metadata = {'sample_indices': []}  # Replace with actual metadata
    return video, metadata

class VideoReasoningGraphExperiment(VideoReasoningExperiment):
    """
    This class represents an experiment that uses a video reasoning graph to reason through a given video and
    a corresponding task. A task is a combination of a task or question and its answer options.
    """

    def __init__(self, config: dict):
        """
        Instances of this class represent specific experiments that are defined with a configuration similar to the
        example configuration in ./config/foundation_models/default_video_describer.yaml.

        :param config: A dictionary specifying the experiment configuration.
        """
        super().__init__(config)

        if torch.cuda.is_available():
            # Get the number of available CUDA devices
            num_devices = torch.cuda.device_count()
            logger.info("Using the following CUDA visible devices:")
            for i in range(num_devices):
                logger.info(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            logger.warning("Note that this framework will use the assigned device indices instead of the real ones.")
        else:
            logger.warning("CUDA is not available. The framework will run on CPU. This is only possible if "
                           "pre-extracted data is used.")

        # Initialize path and dir parameters
        self.videos_path = config.get("videos_path")
        self.tasks_path = config.get("tasks_path")
        self.answers_path = config.get("answers_path")
        self.predictions_path = os.path.join(self.experiment_path, "predictions.json")
        self.conclusions_path = os.path.join(self.experiment_path, "conclusions.json")
        self.decisions_path = os.path.join(self.experiment_path, "decisions.json")
        self.accuracy_path = os.path.join(self.experiment_path, "accuracy.json")
        self.states_path = os.path.join(self.experiment_path, "states")
        self.resume_path = config.get("resume_path", None)
        self.iterate_through_videos = config.get("iterate_through_videos", False)

        # Create experiment paths and files
        self._create_experiment_paths_and_files(config)

        # Initialize video-related parameters
        self.sample_rate = config.get("sample_rate", 1)
        self.start_video = config.get("start_video", 0)
        self.end_video = config.get("end_video", 10 ** 9)

        # Initialize subset indices if given
        self.subset_indices = config.get("subset_indices", None)

        # Initialize video reasoning graph controller parameters
        controller_config = config.get("controller", {})
        self.max_depth = controller_config.get("max_depth", 10)

        # Initialize random seed
        self.random_seed = config.get("random_seed", 42)
        self.reset_seed_for_each_function = config.get("reset_seed_for_each_function", False)
        self.reset_seed_for_each_video = config.get("reset_seed_for_each_video", False)

        # Initialize node state configurations
        self.spatial_node_state_config = config.get("spatial_node_state_config", {})
        self.temporal_node_state_config = config.get("temporal_node_state_config", {})

        # Initialize lexical representation
        self.lexical_representation = config.get("lexical_representation", "")

        # Call initialization methods
        self._initialize_operations(config.get("operations", {}))
        self._initialize_api(config.get("api", {}))

        # Load data
        self.data = load_data(
            answer_path=self.answers_path,
            task_path=self.tasks_path,
            normalize=True
        )

        # If experiment is not new, read existing predictions
        if not self._is_new_experiment():
            self.predictions = read_json_file(file_path=self.predictions_path)
        else:
            self.predictions = {}

    def _is_new_experiment(self):
        """
        Check if the experiment is new based on the existence of the predictions file.
        """
        return not os.path.exists(self.predictions_path)

    def _create_experiment_paths_and_files(self, config):
        """
        Create experiment paths and files.

        :param config: Configuration dictionary.
        """
        if not self.resume_path:
            logger.info("No resume path specified, starting new experiment...")

            paths = [
                self.predictions_path,
                self.conclusions_path,
                self.decisions_path
            ]

            # Create predictions, conclusions, and decisions files if they do not exist
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    write_json_file(data={}, file_path=path)
                    logger.info(f"Created json file at {path}")

            logger.info(f"Starting experiment, saving predictions at {self.predictions_path}")
        else:
            logger.info("Resume path specified, resuming experiment...")

            resume_path = config["resume_path"]
            existing_predictions_path = os.path.join(resume_path, "predictions.json")
            existing_conclusions_path = os.path.join(resume_path, "conclusions.json")
            existing_decisions_path = os.path.join(resume_path, "decisions.json")
            existing_states_path = os.path.join(resume_path, "states")

            # Copy states
            if os.path.exists(existing_states_path):
                shutil.copytree(existing_states_path, self.states_path, dirs_exist_ok=True)
                logger.info(f"Copied states from {existing_states_path} to {self.states_path}")
            else:
                logger.warning(f"No states folder found in resume path: {existing_states_path}")

            # Copy json files
            json_files = {
                "predictions.json": self.predictions_path,
                "conclusions.json": self.conclusions_path,
                "decisions.json": self.decisions_path
            }

            for src, dst in json_files.items():
                src_file = os.path.join(resume_path, src)
                if os.path.exists(src_file):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src_file, dst)
                    logger.info(f"Copied {src} from {resume_path} to {dst}")
                else:
                    logger.warning(f"{src} does not exist in resume path: {resume_path}")

    def _initialize_operations(self, operations_config):
        """
        Initialize operations based on the operations configuration.

        :param operations_config: Dictionary containing operations configuration.
        """
        available_operations = {
            # Structural operations
            "Expand": Expand,
            "DPCKNNExpand": DPCKNNExpand,
            "Wait": Wait,
            "Merge": Merge,
            # State operations
            "DeriveNodeStates": DeriveNodeStates,
            "DeriveRootNodeState": DeriveRootNodeState,
            "Ranking": Ranking,
            "NoRatings": NoRatings,
            "SummaryConfidenceRating": SummaryConfidenceRating,
            "AnswerabilityRating": AnswerabilityRating,
            # Decision operations
            "NoDecision": NoDecision,
            "PairwiseHammingDistanceDecision": PairwiseHammingDistanceDecision,
            # Conclusion operations
            "OptionsCandidateBasedOnConcatenatedLexicalStates": OptionsCandidateBasedOnConcatenatedLexicalStates,
            "HammingSimilarityOfOptionsRankingsPerNodeConclusion": HammingSimilarityOfOptionsRankingsPerNodeConclusion,
            "OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates": OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates,
            "MajorityVoteOfRankingsConclusion": MajorityVoteOfRankingsConclusion,
            "NoConclusion": NoConclusion,
            "IterativeMergeConclusion": IterativeMergeConclusion
        }

        # Initialize all operations
        for op_key, op_class in available_operations.items():
            op_config = operations_config.get(op_key, {})
            if not op_config:
                logger.warning(f"Operation config for '{op_key}' is missing. Skipping initialization.")
                continue

            op_class_name = op_config.get("class")
            op_params = op_config.get("params", {})

            if op_class_name not in available_operations:
                logger.error(f"Operation class '{op_class_name}' is not recognized.")
                continue

            operation_instance = available_operations[op_class_name](**op_params)
            setattr(self, f"{op_key.lower()}_operation", operation_instance)
            logger.info(f"Initialized operation '{op_key}' with class '{op_class_name}'.")

        # Specifically handle decision and conclusion operations which might require other operations as inputs
        # Assuming 'Decision' and 'Conclusion' are keys in operations_config
        decision_config = operations_config.get("decision", {})
        if decision_config:
            decision_class_name = decision_config.get("class")
            decision_params = decision_config.get("params", {})
            decision_class = available_operations.get(decision_class_name)
            if decision_class:
                self.decision = decision_class(
                    expand_operation=self.expand_operation,
                    merge_operation=self.merge_operation,
                    wait_operation=self.wait_operation,
                    **decision_params
                )
                logger.info(f"Initialized 'decision' operation with class '{decision_class_name}'.")
            else:
                logger.error(f"Decision class '{decision_class_name}' is not recognized.")

        conclusion_config = operations_config.get("conclusion", {})
        if conclusion_config:
            conclusion_class_name = conclusion_config.get("class")
            conclusion_params = conclusion_config.get("params", {})
            conclusion_class = available_operations.get(conclusion_class_name)
            if conclusion_class:
                self.conclusion = conclusion_class(**conclusion_params)
                logger.info(f"Initialized 'conclusion' operation with class '{conclusion_class_name}'.")
            else:
                logger.error(f"Conclusion class '{conclusion_class_name}' is not recognized.")

    def _initialize_api(self, api_config):
        """
        Initialize the API based on the API configuration.

        :param api_config: Dictionary containing API configuration.
        """
        get_completion_from_text_config = api_config.get("get_completion_from_text", {})
        llm_class = get_completion_from_text_config.get("llm_class", "")
        if "openai" in llm_class.lower():
            self.random_seed = get_completion_from_text_config.get("seed", self.random_seed)

        self.api = API(
            load_models_only_when_needed=api_config.get("load_models_only_when_needed", True),
            get_object_detections_from_video_clip_and_text_config=api_config.get(
                "get_object_detections_from_video_clip_and_text", {}
            ),
            get_action_captions_from_video_clip_config=api_config.get("get_action_captions_from_video_clip", {}),
            get_summary_from_noisy_perceptive_data_config=api_config.get("get_summary_from_noisy_perceptive_data", {}),
            get_completion_from_text_config=get_completion_from_text_config,
            get_unspecific_objects_from_video_clip_config=api_config.get("get_unspecific_objects_from_video_clip", {}),
            get_specific_objects_from_video_clip_config=api_config.get("get_specific_objects_from_video_clip", {}),
            random_seed=self.random_seed,
            reset_seed_for_each_function=self.reset_seed_for_each_function
        )
        logger.info("API has been initialized.")

    def _train(self):
        logger.error("This framework does currently not support training experiments.")
        raise NotImplementedError("Training function is currently invalid.")

    def _test(self):
        logger.error("This framework does currently not support testing experiments.")
        raise NotImplementedError("Testing function is currently invalid.")

    def _eval(self):
        logger.info("Starting evaluation experiment...")
        logger.info(f"Starting video reasoning for videos {self.start_video} to {self.end_video} "
                    f"(i.e. {self.end_video - self.start_video} videos in total)...")

        # Measure the total execution time
        start_time_total = time.time()

        # Counters for accuracy
        num_samples_correct = 0
        num_samples_total = 0

        # Set to remember visited video ids
        visited_video_ids = set()

        # Iterate through the data
        for video_count, item in enumerate(self.data):
            # Measure execution time per video
            start_time = time.time()

            # Stop if the number of videos is reached
            if video_count >= self.end_video:
                logger.info(f"Reached end of specified video range {self.start_video} to {self.end_video}.")
                break

            video_id = item["video_id"]
            question = item["question"]
            options = item["options"]

            # Skip videos which are not in the specified video range
            if video_count < self.start_video:
                logger.info(f"Skipping video {video_id} because of specified video range...")
                continue

            # Skip videos which have already been processed if self.iterate_through_videos is True
            if self.iterate_through_videos and video_id in visited_video_ids:
                logger.info(f"Skipping video {video_id} because it has already been processed "
                            f"and self.iterate_through_videos is set to True...")
                continue

            # Skip videos which are not in the specified subset indices
            if self.subset_indices is not None and video_count not in self.subset_indices:
                logger.info(f"Skipping video {video_id} because it is not in the specified subset indices...")
                continue

            # Remember the visited video id
            visited_video_ids.add(video_id)

            # Clear caches
            torch.cuda.empty_cache()
            importlib.invalidate_caches()

            # (Re-)set the seed if specified or if it is the first video
            if self.reset_seed_for_each_video or video_count == self.start_video:
                seed = self.random_seed + video_count
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"Random seed has been reset to {seed} to ensure reproducibility and comparability.")

            logger.info(f"Processing video {video_id}: {question}, options: {options}")

            # Get video path
            video_path = next(
                (os.path.join(self.videos_path, f"{video_id}{ext}") for ext in ['.mp4', '.mkv', '.avi']
                 if os.path.exists(os.path.join(self.videos_path, f"{video_id}{ext}"))), None)
            if video_path is None:
                logger.warning(f"No video file found for video_id {video_id} with supported formats ['.mp4', '.mkv', '.avi']")
                continue

            # Create video data from video path
            video, metadata = create_video_data_from_video_path(video_path=video_path, sample_rate=self.sample_rate)
            logger.debug(f"Original sampled indices: {metadata['sample_indices']}")

            # Transform video to video_clip
            video_clip = VideoClip.from_metadata(video, metadata)
            logger.info(f"Loaded and initialized video data from {video_path} with sample rate "
                        f"{self.sample_rate} and {len(video_clip)} frames.")

            # Create the states save path for the current video
            states_save_path = os.path.join(self.states_path, video_id, slugify(question))
            os.makedirs(states_save_path, exist_ok=True)

            # Initialize the task
            task = Task(question=question, options=options)

            # Initialize NodeState and other classes
            spatial_node_state = SpatialNodeState(
                video_clip=video_clip,
                task=task,
                use_action_captions=self.spatial_node_state_config.get("use_action_captions", False),
                use_object_detections=self.spatial_node_state_config.get("use_object_detections", False),
                use_action_captions_summary=self.spatial_node_state_config.get("use_action_captions_summary", False),
                use_object_detections_summary=self.spatial_node_state_config.get("use_object_detections_summary", False),
                lexical_representation=self.lexical_representation
            )
            temporal_node_state = TemporalNodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                use_foreground=self.temporal_node_state_config.get("use_foreground", False),
                use_relevance=self.temporal_node_state_config.get("use_relevance", False),
                use_salience=self.temporal_node_state_config.get("use_salience", False),
                use_temporal_grounding_summary=self.temporal_node_state_config.get("use_temporal_grounding_summary", False)
            )
            root_state = NodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                spatial_node_state=spatial_node_state,
                temporal_node_state=temporal_node_state
            )

            # Initialize Node and Graph
            root_node = Node(state=root_state)
            video_reasoning_graph = Graph(root_node=root_node)

            # Initialize the video reasoning controller
            controller = VideoReasoningController(
                video_reasoning_graph=video_reasoning_graph,
                task=task,
                api=self.api,
                states_save_path=states_save_path,
                expand_operation=self.expand_operation,
                merge_operation=self.merge_operation,
                derive_node_state_operation=self.derive_node_state_operation,
                derive_universal_state_operation=self.derive_universal_state_operation,
                rating=self.rating,
                decision=self.decision,
                conclusion=self.conclusion,
                max_depth=self.max_depth,
                predictions_save_path=self.predictions_path,
                conclusions_save_path=self.conclusions_path,
                decisions_save_path=self.decisions_path
            )

            # Execute the video reasoning
            choice = controller.single_choice_reasoning()

            # Record the correct answers and total number of iterations
            num_samples_total += 1
            if choice == item['answer']:
                num_samples_correct += 1

            # Calculate and log the accuracy
            accuracy = num_samples_correct / num_samples_total
            logger.info(f"Current accuracy: {accuracy:.2f}")

            # Measure execution time per video
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Execution time for video {video_id}: {execution_time:.2f} seconds")

        # Measure the total execution time
        end_time_total = time.time()
        total_time = end_time_total - start_time_total
        logger.info(f"Total execution time: {total_time:.2f} seconds")

        # Evaluate overall predictions
        self._evaluate_predictions()

    def _evaluate_predictions(self):
        """
        Evaluate the predictions iteration-wise and calculate accuracies.
        """
        logger.info("Evaluating predictions...")

        all_predictions = read_json_file(file_path=self.predictions_path)
        num_correct_per_iteration = []
        total_predictions_per_iteration = []
        accuracy_per_iteration = []

        for iteration, (video_id, video_predictions) in enumerate(all_predictions.items(), start=1):
            num_correct = 0
            total_predictions = 0
            for question, prediction in video_predictions.items():
                # Find the corresponding data entry in self.data
                corresponding_data_entry = next(
                    (item for item in self.data if item['video_id'] == video_id and item['question'] == question),
                    None
                )
                if not corresponding_data_entry:
                    logger.warning(f"No corresponding data entry found for video_id {video_id} and question '{question}'.")
                    continue
                ground_truth = corresponding_data_entry['answer']
                if prediction == ground_truth:
                    num_correct += 1
                total_predictions += 1

            if total_predictions == 0:
                # Assign the last known correct count if total predictions are zero
                if num_correct_per_iteration:
                    num_correct = num_correct_per_iteration[-1]
                    total_predictions = total_predictions_per_iteration[-1]
                else:
                    num_correct = 0
                    total_predictions = 0

            num_correct_per_iteration.append(num_correct)
            total_predictions_per_iteration.append(total_predictions)
            accuracy = (num_correct / total_predictions) if total_predictions > 0 else 0
            accuracy_per_iteration.append(accuracy)
            logger.info(f"Iteration {iteration}: {num_correct} correct predictions out of {total_predictions}, Accuracy: {accuracy:.2f}")

        # Save the accuracies
        write_json_file(accuracy_per_iteration, self.accuracy_path)
        logger.info(f"Accuracies saved to {self.accuracy_path}")

###########################################################################################################################################################################3
###########################################################################################################################################################################3
###########################################################################################################################################################################3

# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.

class VideoReasoningGraphExperiment(VideoReasoningExperiment):
    """
    This class represents an experiment that uses a video reasoning graph to reason through a given video and
    a corresponding task. A task is a combination of a task or question and its answer options.
    """

    def __init__(self, config: dict):
        """
        Instances of this class represent specific experiments that are defined with a configuration similar to the
        example configuration in ./config/foundation_models/default_video_describer.yaml.

        :param config: A dictionary specifying the experiment configuration.
        """
        super().__init__(config)

        if torch.cuda.is_available():
            # Get the number of available CUDA devices
            num_devices = torch.cuda.device_count()
            logger.info("Using the following CUDA visible devices:")
            for i in range(num_devices):
                logger.info(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            logger.warning("Note that this framework will use the assigned device indices instead of the real ones.")
        else:
            logger.warning("CUDA is not available. The framework will run on CPU. This is only possible if "
                           "pre-extracted data is used.")

        # Initialize path and dir parameters
        self.videos_path = config.get("videos_path")
        self.tasks_path = config.get("tasks_path")
        self.answers_path = config.get("answers_path")
        self.predictions_path = os.path.join(self.experiment_path, "predictions.json")
        self.conclusions_path = os.path.join(self.experiment_path, "conclusions.json")
        self.decisions_path = os.path.join(self.experiment_path, "decisions.json")
        self.accuracy_path = os.path.join(self.experiment_path, "accuracy.json")
        self.states_path = os.path.join(self.experiment_path, "states")
        self.resume_path = config.get("resume_path", None)
        self.iterate_through_videos = config.get("iterate_through_videos", False)

        # Create experiment paths and files
        self._create_experiment_paths_and_files(config)

        # Initialize video-related parameters
        self.sample_rate = config.get("sample_rate", 1)
        self.start_video = config.get("start_video", 0)
        self.end_video = config.get("end_video", 10 ** 9)

        # Initialize subset indices if given
        self.subset_indices = config.get("subset_indices", None)

        # Initialize video reasoning graph controller parameters
        controller_config = config.get("controller", {})
        self.max_depth = controller_config.get("max_depth", 10)

        # Initialize random seed
        self.random_seed = config.get("random_seed", 42)
        self.reset_seed_for_each_function = config.get("reset_seed_for_each_function", False)
        self.reset_seed_for_each_video = config.get("reset_seed_for_each_video", False)

        # Initialize state parameters
        self.spatial_node_state_config = config.get("state", {}).get("spatial_node_state", {})
        self.temporal_node_state_config = config.get("state", {}).get("temporal_node_state", {})
        self.lexical_representation = config.get("state", {}).get("lexical_representation", "")

        # Initialize the operations
        self._initialize_operations(config.get("operations", {}))

        # Initialize the API
        self._initialize_api(config.get("api", {}))

        # Initialize data
        self.data = load_data(
            answers_path=self.answers_path,
            tasks_path=self.tasks_path,
            normalize=config.get("normalize_data", False)
        )
        logger.info(f"Loaded data from {self.answers_path} and {self.tasks_path}.")

        # Load the predictions (empty if starting new experiment)
        if not self._is_new_experiment():
            self.predictions = read_json_file(file_path=self.predictions_path)
            logger.info(f"Loaded predictions from {self.predictions_path}.")
        else:
            self.predictions = {}
            logger.info("No existing predictions found. Starting fresh.")

        logger.info("Initialized VideoReasoningGraphExperiment")

    def _create_experiment_paths_and_files(self, config):
        """
        Create experiment paths and files.

        :param config: Configuration dictionary.
        """
        if not self.resume_path:
            logger.info("No resume path specified, starting new experiment...")

            paths = [
                self.predictions_path,
                self.conclusions_path,
                self.decisions_path
            ]

            # Create predictions, conclusions, and decisions files if they do not exist
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    write_json_file(data={}, file_path=path)
                    logger.info(f"Created JSON file at {path}")

            logger.info(f"Starting experiment, saving predictions at {self.predictions_path}")
        else:
            logger.info("Resume path specified, resuming experiment...")

            resume_path = config["resume_path"]
            existing_predictions_path = os.path.join(resume_path, "predictions.json")
            existing_conclusions_path = os.path.join(resume_path, "conclusions.json")
            existing_decisions_path = os.path.join(resume_path, "decisions.json")
            existing_states_path = os.path.join(resume_path, "states")

            # Verify existence of required files/directories
            if not os.path.exists(existing_predictions_path):
                err_msg = f"Predictions file at {existing_predictions_path} does not exist."
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            if not os.path.exists(existing_states_path):
                err_msg = f"States directory at {existing_states_path} does not exist."
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            # Copy states
            shutil.copytree(src=existing_states_path, dst=self.states_path, dirs_exist_ok=True)
            logger.info(f"Copied states from {existing_states_path} to {self.states_path}")

            # Copy JSON files
            json_files = {
                "predictions.json": self.predictions_path,
                "conclusions.json": self.conclusions_path,
                "decisions.json": self.decisions_path
            }

            for src_file, dst_file in json_files.items():
                src = os.path.join(resume_path, src_file)
                if os.path.exists(src):
                    shutil.copyfile(src, dst_file)
                    logger.info(f"Copied {src_file} from {resume_path} to {dst_file}")
                else:
                    logger.warning(f"{src_file} does not exist in resume path: {resume_path}")

            logger.info(f"Copied existing experiment from {resume_path} to {self.experiment_path}")
            logger.info(f"Resuming experiment, skipping videos for which predictions already "
                        f"have been made in the past at {resume_path}")

    def _initialize_operations(self, operations_config):
        """
        Initialize operations based on the operations configuration.

        :param operations_config: Dictionary containing operations configuration.
        """
        available_operations = {
            # Structural operations
            "Expand": Expand,
            "DPCKNNExpand": DPCKNNExpand,
            "Wait": Wait,
            "Merge": Merge,
            # State operations
            "DeriveNodeStates": DeriveNodeStates,
            "DeriveRootNodeState": DeriveRootNodeState,
            "Ranking": Ranking,
            "NoRatings": NoRatings,
            "SummaryConfidenceRating": SummaryConfidenceRating,
            "AnswerabilityRating": AnswerabilityRating,
            # Decision operations
            "NoDecision": NoDecision,
            "PairwiseHammingDistanceDecision": PairwiseHammingDistanceDecision,
            # Conclusion operations
            "OptionsCandidateBasedOnConcatenatedLexicalStates": OptionsCandidateBasedOnConcatenatedLexicalStates,
            "HammingSimilarityOfOptionsRankingsPerNodeConclusion": HammingSimilarityOfOptionsRankingsPerNodeConclusion,
            "OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates": OptionsCandidateBasedOnAnswerableConcatenatedLexicalStates,
            "MajorityVoteOfRankingsConclusion": MajorityVoteOfRankingsConclusion,
            "NoConclusion": NoConclusion,
            "IterativeMergeConclusion": IterativeMergeConclusion
        }

        # Initialize all operations
        for op_key, op_class in available_operations.items():
            op_config = operations_config.get(op_key, {})
            if not op_config:
                logger.warning(f"Operation config for '{op_key}' is missing. Skipping initialization.")
                continue

            op_class_name = op_config.get("class")
            op_params = op_config.get("params", {})

            # Retrieve the class from available_operations
            operation_class = available_operations.get(op_class_name)
            if not operation_class:
                logger.error(f"Operation class '{op_class_name}' for '{op_key}' is not recognized.")
                continue

            # Initialize the operation instance
            operation_instance = operation_class(**op_params)
            setattr(self, f"{op_key.lower()}_operation", operation_instance)
            logger.info(f"Initialized operation '{op_key}' with class '{op_class_name}'.")

        # Specifically handle decision and conclusion operations which might require other operations as inputs
        # Assuming 'Decision' and 'Conclusion' are keys in operations_config
        decision_config = operations_config.get("Decision", {})
        if decision_config:
            decision_class_name = decision_config.get("class")
            decision_params = decision_config.get("params", {})
            decision_class = available_operations.get(decision_class_name)
            if decision_class:
                self.decision = decision_class(
                    expand_operation=self.expand_operation,
                    merge_operation=self.merge_operation,
                    wait_operation=self.wait_operation,
                    **decision_params
                )
                logger.info(f"Initialized 'decision' operation with class '{decision_class_name}'.")
            else:
                logger.error(f"Decision class '{decision_class_name}' is not recognized.")

        conclusion_config = operations_config.get("Conclusion", {})
        if conclusion_config:
            conclusion_class_name = conclusion_config.get("class")
            conclusion_params = conclusion_config.get("params", {})
            conclusion_class = available_operations.get(conclusion_class_name)
            if conclusion_class:
                self.conclusion = conclusion_class(**conclusion_params)
                logger.info(f"Initialized 'conclusion' operation with class '{conclusion_class_name}'.")
            else:
                logger.error(f"Conclusion class '{conclusion_class_name}' is not recognized.")

    def _initialize_api(self, api_config):
        """
        Initialize the API based on the API configuration.

        :param api_config: Dictionary containing API configuration.
        """
        get_completion_from_text_config = api_config.get("get_completion_from_text", {})
        llm_class = get_completion_from_text_config.get("llm_class", "")
        if "openai" in llm_class.lower():
            self.random_seed = get_completion_from_text_config.get("seed", self.random_seed)

        self.api = API(
            load_models_only_when_needed=api_config.get("load_models_only_when_needed", True),
            get_object_detections_from_video_clip_and_text_config=api_config.get(
                "get_object_detections_from_video_clip_and_text", {}
            ),
            get_action_captions_from_video_clip_config=api_config.get("get_action_captions_from_video_clip", {}),
            get_temporal_grounding_from_video_clip_and_text_config=api_config.get(
                "get_temporal_grounding_from_video_clip_and_text", {}
            ),
            get_summary_from_noisy_perceptive_data_config=api_config.get(
                "get_summary_from_noisy_perceptive_data", {}
            ),
            get_completion_from_text_config=get_completion_from_text_config,
            get_unspecific_objects_from_video_clip_config=api_config.get("get_unspecific_objects_from_video_clip", {}),
            get_specific_objects_from_video_clip_config=api_config.get("get_specific_objects_from_video_clip", {}),
            random_seed=self.random_seed,
            reset_seed_for_each_function=self.reset_seed_for_each_function
        )
        logger.info("API has been initialized.")

    def _train(self):
        logger.error("This framework does currently not support training experiments.")
        raise NotImplementedError("Training function is currently invalid.")

    def _test(self):
        logger.error("This framework does currently not support testing experiments.")
        raise NotImplementedError("Testing function is currently invalid.")

    def _eval(self):
        logger.info("Starting evaluation experiment...")
        logger.info(f"Starting video reasoning for videos {self.start_video} to {self.end_video} "
                    f"(i.e. {self.end_video - self.start_video} videos in total)...")

        # Measure the total execution time
        start_time_total = time.time()

        # Counters for accuracy
        num_samples_correct = 0
        num_samples_total = 0

        # Set to remember visited video ids
        visited_video_ids = set()

        # Iterate through the data
        for video_count, item in enumerate(self.data):
            # Measure execution time per video
            start_time = time.time()

            # Stop if the number of videos is reached
            if video_count >= self.end_video:
                logger.info(f"Reached end of specified video range {self.start_video} to {self.end_video}.")
                break

            video_id = item["video_id"]
            question = item["question"]
            options = item["options"]

            # Skip videos which are not in the specified video range
            if video_count < self.start_video:
                logger.info(f"Skipping video {video_id} because of specified video range...")
                continue

            # Skip videos which have already been processed if self.iterate_through_videos is True
            if self.iterate_through_videos and video_id in visited_video_ids:
                logger.info(f"Skipping video {video_id} because it has already been processed "
                            f"and self.iterate_through_videos is set to True...")
                continue

            # Skip videos which are not in the specified subset indices
            if self.subset_indices is not None and video_count not in self.subset_indices:
                logger.info(f"Skipping video {video_id} because it is not in the specified subset indices...")
                continue

            # Remember the visited video id
            visited_video_ids.add(video_id)

            # Clear caches
            torch.cuda.empty_cache()
            importlib.invalidate_caches()

            # (Re-)set the seed if specified or if it is the first video
            if self.reset_seed_for_each_video or video_count == self.start_video:
                seed = self.random_seed + video_count
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"Random seed has been reset to {seed} to ensure reproducibility and comparability.")

            logger.info(f"Starting video reasoning for video {video_id}...")
            logger.info(f"Question: {question}.")
            logger.info(f"Options: {options}")

            # Load and initialize video data
            video_path = next(
                (os.path.join(self.videos_path, f"{video_id}{ext}") for ext in ['.mp4', '.mkv', '.avi']
                 if os.path.exists(os.path.join(self.videos_path, f"{video_id}{ext}"))), None)
            if video_path is None:
                logger.warning(f"No video file found for video_id {video_id} with supported formats ['.mp4', '.mkv', '.avi']")
                continue

            video, metadata = create_video_data_from_video_path(video_path=video_path, sample_rate=self.sample_rate)
            logger.debug(f"Original sampled indices: {metadata['sample_indices']}")

            # Transform video to video_clip
            video_clip = VideoClip.from_metadata(video, metadata)
            logger.info(f"Loaded and initialized video data from {video_path} with sample rate "
                        f"{self.sample_rate} and {len(video_clip)} frames.")

            # Create the states save path for the current video
            states_save_path = os.path.join(self.states_path, video_id, slugify(question))
            os.makedirs(states_save_path, exist_ok=True)

            # Initialize the task
            task = Task(question=question, options=options)

            # Initialize NodeState and other classes
            spatial_node_state = SpatialNodeState(
                video_clip=video_clip,
                task=task,
                use_action_captions=self.spatial_node_state_config.get("use_action_captions", False),
                use_object_detections=self.spatial_node_state_config.get("use_object_detections", False),
                use_action_captions_summary=self.spatial_node_state_config.get("use_action_captions_summary", False),
                use_object_detections_summary=self.spatial_node_state_config.get("use_object_detections_summary", False),
                lexical_representation=self.lexical_representation
            )
            temporal_node_state = TemporalNodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                use_foreground=self.temporal_node_state_config.get("use_foreground", False),
                use_relevance=self.temporal_node_state_config.get("use_relevance", False),
                use_salience=self.temporal_node_state_config.get("use_salience", False),
                use_temporal_grounding_summary=self.temporal_node_state_config.get("use_temporal_grounding_summary", False)
            )
            root_state = NodeState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                spatial_node_state=spatial_node_state,
                temporal_node_state=temporal_node_state
            )

            # Initialize Node and Graph
            root_node = Node(state=root_state)
            video_reasoning_graph = Graph(root_node=root_node)

            # Initialize the video reasoning controller
            video_reasoning_controller = VideoReasoningController(
                video_reasoning_graph=video_reasoning_graph,
                task=task,
                api=self.api,
                states_save_path=states_save_path,
                expand_operation=self.expand_operation,
                merge_operation=self.merge_operation,
                derive_node_state_operation=self.derive_node_state_operation,
                derive_universal_state_operation=self.derive_root_node_state_operation,
                rating=self.derive_rating_operation,
                decision=self.decision,
                conclusion=self.conclusion,
                max_depth=self.max_depth,
                predictions_save_path=self.predictions_path,
                conclusions_save_path=self.conclusions_path,
                decisions_save_path=self.decisions_path
            )

            # Execute the video reasoning, choose one of the options as answer
            choice = video_reasoning_controller.single_choice_reasoning()

            # Remember the number of correct answers
            if choice == item["answer"]:
                num_samples_correct += 1

            # Remember the number of total samples
            num_samples_total += 1

            # Calculate the current accuracy and log it
            current_accuracy = num_samples_correct / num_samples_total
            logger.info(f"The accuracy after processing {num_samples_total} videos is {current_accuracy:.2f}.")

            # Measure execution time per video
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Reasoning for video {video_id} took {execution_time:.2f} seconds.")

        logger.info("Evaluating the predicted answers...")

        # Evaluate the predictions iteration-wise
        self._evaluate_predictions()

        # Measure the total execution time
        end_time_total = time.time()
        total_time = end_time_total - start_time_total
        logger.info(f"Finished evaluation experiment in {total_time:.2f} seconds.")

    def _evaluate_predictions(self):
        """
        Evaluate the predictions iteration-wise and calculate accuracies.
        """
        logger.info("Evaluating predictions...")

        all_predictions = read_json_file(file_path=self.predictions_path)
        num_correct_per_iteration = [0] * self.max_depth
        num_total = [0] * self.max_depth

        for video_id, video_predictions in all_predictions.items():
            # There can be more than one prediction per video
            # (since datasets like NExT-QA have multiple questions per video)
            for question, question_prediction in video_predictions.items():
                # Get the data entry corresponding to the video_id and question
                corresponding_data_entries = [item for item in self.data if
                                              item["video_id"] == video_id and
                                              item["question"] == question]
                if not corresponding_data_entries:
                    logger.warning(f"No corresponding data entry found for video_id {video_id} and question '{question}'.")
                    continue
                corresponding_data_entry = corresponding_data_entries[0]

                # Get the ground truth
                ground_truth = corresponding_data_entry["answer"]

                # Compare the ground truth with the prediction for each iteration
                for iteration, prediction in question_prediction.items():
                    logger.debug(f"Comparison of video {video_id} and question '{question}': "
                                 f"GT: {ground_truth} - Prediction: {prediction}")

                    # Get the index for this iteration (assuming iterations are 1-based)
                    try:
                        index = int(iteration) - 1
                    except ValueError:
                        logger.error(f"Invalid iteration number '{iteration}' for video {video_id}, question '{question}'.")
                        continue

                    if 0 <= index < self.max_depth:
                        if prediction == ground_truth:
                            num_correct_per_iteration[index] += 1
                        num_total[index] += 1
                    else:
                        logger.warning(f"Iteration {iteration} out of bounds for max_depth {self.max_depth}.")

        # Fill up the rest of the iterations with the last valid value if necessary
        for i in range(1, self.max_depth):
            if num_correct_per_iteration[i] == 0 and num_total[i] == 0 and num_total[i-1] > 0:
                num_correct_per_iteration[i] = num_correct_per_iteration[i-1]
                num_total[i] = num_total[i-1]

        # Calculate the accuracy per iteration
        accuracy_per_iteration = {}
        for i in range(self.max_depth):
            iteration_num = str(i + 1)
            num_correct = num_correct_per_iteration[i]
            total = num_total[i]
            accuracy = num_correct / total if total > 0 else "N/A"
            accuracy_per_iteration[iteration_num] = {
                "num_correct": num_correct,
                "num_total": total,
                "accuracy": accuracy
            }
            logger.info(f"Iteration {iteration_num}: {num_correct} correct predictions out of {total}, "
                        f"Accuracy: {accuracy if accuracy != 'N/A' else 'N/A'}")

        # Save the accuracies
        write_json_file(data=accuracy_per_iteration, file_path=self.accuracy_path)
        logger.info(f"Accuracies saved to {self.accuracy_path}")