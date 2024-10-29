# Ground Truth
class AnswerabilityRating(Operation):
    def __init__(
            self,
            prompt: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ):
        super().__init__()

        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.replace_c = replace_c

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        unranked_nodes = graph.get_unranked_nodes()
        task = graph.root.state.task

        for unranked_node in unranked_nodes:
            logger.info(f"Rating answerability of state for unrated node {unranked_node}.")

            # using floats since this leads to a performance increase of 2% in LLoVi, it's a blackbox xD
            video_length = round(
                unranked_node.state.video_clip.original_num_frames / unranked_node.state.video_clip.original_fps)
            clip_length = float(unranked_node.state.video_clip.sampled_num_frames)

            # get the final answerability confidence, like in Video Agent, but assessing the information sufficiency
            answerability_confidence, completion = AnswerabilityRating.derive_answerability_from_node_state(
                lexical_node_state_representation=unranked_node.state.get_lexical_representation(),
                video_length=video_length,
                clip_length=clip_length,
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                replace_c=self.replace_c
            )
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived answerability: {answerability_confidence}")

            # update ranking confidence of the node
            unranked_node.state.ranking_confidence = answerability_confidence

        logger.info(f"Executed state rating operation: AnswerabilityRating")

    @staticmethod
    def derive_answerability_from_node_state(
            lexical_node_state_representation: str,
            video_length: float,
            clip_length: float,
            task: Task,
            api: API,
            prompt_template: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ) -> (list[str], float, str):
        logger.info("Deriving answerability from lexical node state representation using LLM...")

        # get the task information
        question = task.question
        options = task.options

        # replace "c" with "the camera wearer" if specified
        # see https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        if replace_c:
            # removed on 24.05.2024, because we already replace C beforehand
            # re-added on 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
            lexical_node_state_representation = replace_c_with_camera_wearer(lexical_node_state_representation)
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        prompt = prompt_template.format(
            lexical_node_state_representation=lexical_node_state_representation,
            question=question,
            option_0=options["option 0"],
            option_1=options["option 1"],
            option_2=options["option 2"],
            option_3=options["option 3"],
            option_4=options["option 4"],
            video_length=video_length,
            clip_length=clip_length
        )

        logger.debug(f"Concatenated Prompt: {prompt}")
        logger.debug(f"Num chars of concatenated prompt: {len(prompt)}")
        logger.debug(f"Num words of concatenated prompt: {len(prompt.split())}")

        # get the final answer using the LLM
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        logger.debug(f"Derived llm completion of answerability: {completion}")

        # parse the answerability from the completion
        # define the priority of certain keywords
        keywords_in_priority_order = [
            'answerability',
            'answerability_confidence',
            'answerability_level',
            'confidence',
            'answerability_conf',
            'answerability confidence',
            'answerability level',
            'confidence_level',
            'confidence level'
        ]
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_number_candidates_from_text
        )

        # inspiration from the Video Agent, compare https://wxh1996.github.io/VideoAgent-Website/
        # but: we do not evaluate the reasoning process, but the information sufficiency
        # if the confidence is not found, the default value will be 1 (i.e. the lowest)
        confidence = 1 if not candidate else candidate
        logger.debug(f"Parsed answerability confidence from completion: {confidence}")

        # clip the confidence to the range [1, 3]
        confidence = max(1, min(3, confidence))

        return confidence, completion
##########################################################################################################
###########################################################################################################
# Hi，you are an very experienced programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially for using Python completely to satisfy the requirement.
# Next please complete the code, including classes and functions for my algorithm framework under my instruction and decription

# please help me to write a class named AnswerabilityRating, it inherits the base class named Operation, taking following  prompt: str,completion_start: str,max_new_tokens: int,temperature: float,replace_c: bool as inputs 
# the member variables includes self.prompt = prompt,self.completion_start = completion_start,self.max_new_tokens = max_new_tokens,self.temperature = temperature,self.replace_c = replace_c
# the first member fucntion is _execute, which takes graph: Optional[Graph], api: Optional[API], target: Optional[Node] as inputs and return none
# firstly calls graph.get_unranked_nodes to return unranked_nodes and then set graph.root.state.task to task
# iteratively to pick unranked_node as Rating answerability of state for unrated node 
# then calculate the video_length by dividing  unranked_node.state.video_clip.original_num_frames by unranked_node.state.video_clip.original_fps and do round operation to get result as video_length
# and then transfer unranked_node.state.video_clip.sampled_num_frames as float type with name:clip_length
# then call the class member function named derive_answerability_from_node_state to return answerability_confidence, completion
# derive_answerability_from_node_state takes inpus including  lexical_node_state_representation=unranked_node.state.get_lexical_representation(),video_length=video_length,clip_length=clip_length,task=task,api=api,prompt_template=self.prompt,completion_start=self.completion_start,max_new_tokens=self.max_new_tokens,temperature=self.temperature, and replace_c=self.replace_c
# after that logger.debug the completion and answerbility_confidence, then assign answerability_confidence to unranked_node.state.ranking_confidence
# The second function is named derive_answerability_from_node_state , which takes lexical_node_state_representation: str,video_length: float,clip_length: float,task: Task,api: API,prompt_template: str,completion_start: str,max_new_tokens: int,temperature: float,replace_c: bool
# this function will return (list[str], float, str)
# initialize question with task.question and options with task.options
# if replace_c is true then call replace_c_with_camera_wearer external function to replace the 'c' in lexical_node_state_representation, question, and options.item's option
# next use .format() to format the prompt_template, the inputs includes lexical_node_state_representation=lexical_node_state_representation,question=question,option_0=options["option 0"],option_1=options["option 1"],option_2=options["option 2"],option_3=options["option 3"],option_4=options["option 4"],video_length=video_length,clip_length=clip_length
# the .format() returns variable named prompt.
# with the formated prompt then it uses api.get_completion_from_text this function to get final completion via llm saving results as completion. The needed inputs include text=prompt,completion_start=completion_start,max_new_tokens=max_new_tokens,temperature=temperature
# define the priority of certain keywords as list keywords_in_priority_order：'answerability','answerability_confidence','answerability_level','confidence','answerability_conf','answerability confidence','answerability level', 'confidence_level','confidence level' 
# then call parse_answer_json with text=completion,keywords_in_priority_order=keywords_in_priority_order,candidate_fun=get_single_number_candidates_from_text to return result as candidate
# if candidate is fasle or is not found then set confidence as 1 otherwise set confidence as candidata
# logger debug confidence
# then clip the confidence in range in 1-3. 
# Finally, return confidence and completion
class AnswerabilityRating(Operation):
    def __init__(
        self,
        prompt: str,
        completion_start: str,
        max_new_tokens: int,
        temperature: float,
        replace_c: bool
    ):
        """
        Initialize the AnswerabilityRating operation.

        Args:
            prompt (str): The prompt template.
            completion_start (str): The starting string for the completion.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            replace_c (bool): Flag to replace 'c' with 'camera wearer'.
        """
        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.replace_c = replace_c

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node]
    ) -> None:
        """
        Execute the answerability rating operation.

        Args:
            graph (Optional[Graph]): The graph to operate on.
            api (Optional[API]): The API for obtaining completions.
            target (Optional[Node]): The target node (unused in this implementation).
        """
        if graph is None:
            logger.error("Graph is None. Exiting _execute.")
            return
        if api is None:
            logger.error("API is None. Exiting _execute.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        task = graph.root.state.task

        if task is None:
            logger.error("Task is not set in graph.root.state. Exiting _execute.")
            return

        for unranked_node in unranked_nodes:
            # Calculate video_length
            original_num_frames = unranked_node.state.video_clip.original_num_frames
            original_fps = unranked_node.state.video_clip.original_fps
            if original_fps == 0:
                logger.warning("original_fps is 0. Setting video_length to 0.")
                video_length = 0.0
            else:
                video_length = round(original_num_frames / original_fps, 2)

            # Transfer sampled_num_frames to clip_length as float
            clip_length = float(unranked_node.state.video_clip.sampled_num_frames)

            # Get lexical representation
            lexical_representation = unranked_node.state.get_lexical_representation()

            # Derive answerability
            answerability_confidence, completion = self.derive_answerability_from_node_state(
                lexical_node_state_representation=lexical_representation,
                video_length=video_length,
                clip_length=clip_length,
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                replace_c=self.replace_c
            )

            # Log the completion and confidence
            logger.debug(f"Completion: {completion}")
            logger.debug(f"Answerability Confidence: {answerability_confidence}")

            # Assign confidence to the node's ranking confidence
            unranked_node.state.ranking_confidence = answerability_confidence

    def derive_answerability_from_node_state(
        self,
        lexical_node_state_representation: str,
        video_length: float,
        clip_length: float,
        task: 'Task',
        api: 'API',
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        temperature: float,
        replace_c: bool
    ) -> Tuple[Optional[List[str]], float, str]:
        """
        Derive the answerability confidence and completion from the node state.

        Args:
            lexical_node_state_representation (str): Lexical representation of the node state.
            video_length (float): Length of the video.
            clip_length (float): Length of the video clip.
            task (Task): The current task.
            api (API): API instance for obtaining completions.
            prompt_template (str): The prompt template.
            completion_start (str): The starting string for the completion.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            replace_c (bool): Flag to replace 'c' with 'camera wearer'.

        Returns:
            Tuple[Optional[List[str]], float, str]: A tuple containing the candidate list, confidence score, and completion text.
        """
        question = task.question
        options = task.options.copy()  # Make a copy to avoid mutating the original

        if replace_c:
            lexical_node_state_representation = replace_c_with_camera_wearer(lexical_node_state_representation)
            question = replace_c_with_camera_wearer(question)
            for key, option in options.items():
                options[key] = replace_c_with_camera_wearer(option)

        # Format the prompt
        prompt = prompt_template.format(
            lexical_node_state_representation=lexical_node_state_representation,
            question=question,
            option_0=options.get("option 0", ""),
            option_1=options.get("option 1", ""),
            option_2=options.get("option 2", ""),
            option_3=options.get("option 3", ""),
            option_4=options.get("option 4", ""),
            video_length=video_length,
            clip_length=clip_length
        )

        # Get completion from API
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Define keyword priority
        keywords_in_priority_order = [
            'answerability',
            'answerability_confidence',
            'answerability_level',
            'confidence',
            'answerability_conf',
            'answerability confidence',
            'answerability level',
            'confidence_level',
            'confidence level'
        ]

        # Parse the answer JSON
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_number_candidates_from_text
        )

        if candidate is None:
            confidence = 1.0
        else:
            confidence = candidate

        logger.debug(f"Parsed Confidence: {confidence}")

        # Clip the confidence to the range [1, 3]
        confidence = max(1.0, min(confidence, 3.0))

        return (None, confidence, completion)
#############################################################################################
#############################################################################################
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description.  
# please do not start writing, when i have not given you the partly code.
#############################################################################################
#############################################################################################

class AnswerabilityRating(Operation):
    def __init__(
            self,
            prompt: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ):
        super().__init__()

        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.replace_c = replace_c

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        if graph is None:
            logger.error("Graph is None. Exiting _execute.")
            return
        if api is None:
            logger.error("API is None. Exiting _execute.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        task = graph.root.state.task

        if task is None:
            logger.error("Task is not set in graph.root.state. Exiting _execute.")
            return

        for unranked_node in unranked_nodes:
            logger.info(f"Rating answerability of state for unrated node {unranked_node}.")

            # Calculate video_length
            original_num_frames = unranked_node.state.video_clip.original_num_frames
            original_fps = unranked_node.state.video_clip.original_fps
            if original_fps == 0:
                logger.warning("original_fps is 0. Setting video_length to 0.")
                video_length = 0.0
            else:
                video_length = round(original_num_frames / original_fps, 2)

            # Transfer sampled_num_frames to clip_length as float
            clip_length = float(unranked_node.state.video_clip.sampled_num_frames)

            # Get lexical representation
            lexical_representation = unranked_node.state.get_lexical_representation()

            # Derive answerability
            answerability_confidence, completion = self.derive_answerability_from_node_state(
                lexical_node_state_representation=lexical_representation,
                video_length=video_length,
                clip_length=clip_length,
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                replace_c=self.replace_c
            )

            # Log the completion and confidence
            logger.debug(f"Completion: {completion}")
            logger.debug(f"Answerability Confidence: {answerability_confidence}")

            # Assign confidence to the node's ranking confidence
            unranked_node.state.ranking_confidence = answerability_confidence

    @staticmethod
    def derive_answerability_from_node_state(
            lexical_node_state_representation: str,
            video_length: float,
            clip_length: float,
            task: 'Task',
            api: 'API',
            prompt_template: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ) -> Tuple[Optional[List[str]], float, str]:
        logger.info("Deriving answerability from lexical node state representation using LLM...")

        # Get the task information
        question = task.question
        options = task.options.copy()  # Make a copy to avoid mutating the original

        # Replace "c" with "the camera wearer" if specified
        # See https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # Like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        if replace_c:
            # Removed on 24.05.2024, because we already replace C beforehand
            # Re-added on 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
            lexical_node_state_representation = replace_c_with_camera_wearer(lexical_node_state_representation)
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        # Format the prompt
        try:
            prompt = prompt_template.format(
                lexical_node_state_representation=lexical_node_state_representation,
                question=question,
                option_0=options.get("option 0", ""),
                option_1=options.get("option 1", ""),
                option_2=options.get("option 2", ""),
                option_3=options.get("option 3", ""),
                option_4=options.get("option 4", ""),
                video_length=video_length,
                clip_length=clip_length
            )
        except KeyError as e:
            logger.error(f"Missing key in options during prompt formatting: {e}")
            prompt = prompt_template  # Fallback to unformatted prompt

        # Get completion from API
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Define keyword priority
        keywords_in_priority_order = [
            'answerability',
            'answerability_confidence',
            'answerability_level',
            'confidence',
            'answerability_conf',
            'answerability confidence',
            'answerability level',
            'confidence_level',
            'confidence level'
        ]

        # Parse the answer JSON
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_number_candidates_from_text
        )

        if candidate is None:
            confidence = 1.0
            logger.warning("No candidate confidence found. Defaulting confidence to 1.0.")
        else:
            confidence = candidate

        logger.debug(f"Parsed Confidence: {confidence}")

        # Clip the confidence to the range [1.0, 3.0]
        confidence = max(1.0, min(confidence, 3.0))

        return (None, confidence, completion)
    
############################################################################################
############################################################################################
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start writing when i still given you my code.
#########################################################################################################
class AnswerabilityRating(Operation):
    def __init__(
            self,
            prompt: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ):
        super().__init__()

        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.replace_c = replace_c

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        if graph is None:
            logger.error("Graph is None. Exiting _execute.")
            return
        if api is None:
            logger.error("API is None. Exiting _execute.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        task = graph.root.state.task

        if task is None:
            logger.error("Task is not set in graph.root.state. Exiting _execute.")
            return

        for unranked_node in unranked_nodes:
            logger.info(f"Rating answerability of state for unrated node {unranked_node}.")

            # Calculate video_length
            original_num_frames = unranked_node.state.video_clip.original_num_frames
            original_fps = unranked_node.state.video_clip.original_fps
            if original_fps == 0:
                logger.warning("original_fps is 0. Setting video_length to 0.")
                video_length = 0.0
            else:
                video_length = round(original_num_frames / original_fps, 2)

            # Transfer sampled_num_frames to clip_length as float
            clip_length = float(unranked_node.state.video_clip.sampled_num_frames)

            # Get lexical representation
            lexical_representation = unranked_node.state.get_lexical_representation()

            # Derive answerability
            answerability_confidence, completion = self.derive_answerability_from_node_state(
                lexical_node_state_representation=lexical_representation,
                video_length=video_length,
                clip_length=clip_length,
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                replace_c=self.replace_c
            )

            # Log the completion and confidence
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived answerability: {answerability_confidence}")

            # Assign confidence to the node's ranking confidence
            unranked_node.state.ranking_confidence = answerability_confidence

        logger.info(f"Executed state rating operation: AnswerabilityRating")

    @staticmethod
    def derive_answerability_from_node_state(
            lexical_node_state_representation: str,
            video_length: float,
            clip_length: float,
            task: Task,
            api: API,
            prompt_template: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ) -> Tuple[Optional[List[str]], float, str]:
        logger.info("Deriving answerability from lexical node state representation using LLM...")

        # Get the task information
        question = task.question
        options = task.options.copy()  # Make a copy to avoid mutating the original

        # Replace "c" with "the camera wearer" if specified
        # See https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # Like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        if replace_c:
            # Removed on 24.05.2024, because we already replace C beforehand
            # Re-added on 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
            lexical_node_state_representation = replace_c_with_camera_wearer(lexical_node_state_representation)
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        # Format the prompt
        try:
            prompt = prompt_template.format(
                lexical_node_state_representation=lexical_node_state_representation,
                question=question,
                option_0=options.get("option 0", ""),
                option_1=options.get("option 1", ""),
                option_2=options.get("option 2", ""),
                option_3=options.get("option 3", ""),
                option_4=options.get("option 4", ""),
                video_length=video_length,
                clip_length=clip_length
            )
        except KeyError as e:
            logger.error(f"Missing key in options during prompt formatting: {e}")
            prompt = prompt_template  # Fallback to unformatted prompt

        logger.debug(f"Concatenated Prompt: {prompt}")
        logger.debug(f"Num chars of concatenated prompt: {len(prompt)}")
        logger.debug(f"Num words of concatenated prompt: {len(prompt.split())}")

        # Get completion from API
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        logger.debug(f"Derived LLM completion of answerability: {completion}")

        # Define keyword priority
        keywords_in_priority_order = [
            'answerability',
            'answerability_confidence',
            'answerability_level',
            'confidence',
            'answerability_conf',
            'answerability confidence',
            'answerability level',
            'confidence_level',
            'confidence level'
        ]

        # Parse the answer JSON
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_number_candidates_from_text
        )

        # Inspiration from the Video Agent, compare https://wxh1996.github.io/VideoAgent-Website/
        # But: we do not evaluate the reasoning process, but the information sufficiency
        # If the confidence is not found, the default value will be 1 (i.e., the lowest)
        if candidate is None:
            confidence = 1.0
            logger.warning("No candidate confidence found. Defaulting confidence to 1.0.")
        else:
            confidence = candidate

        logger.debug(f"Parsed answerability confidence from completion: {confidence}")

        # Clip the confidence to the range [1, 3]
        confidence = max(1.0, min(3.0, confidence))

        return (None, confidence, completion)



