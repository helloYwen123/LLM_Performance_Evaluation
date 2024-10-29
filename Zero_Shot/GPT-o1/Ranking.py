# Ground Truth
class Ranking(Operation):
    def __init__(self, prompt: str, completion_start: str, max_new_tokens: int, parse_confidence: bool = False):
        super().__init__()

        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.parse_confidence = parse_confidence

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        unranked_nodes = graph.get_unranked_nodes()
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        for unranked_node in unranked_nodes:
            logger.info(f"Rating state for unrated node {unranked_node}.")

            # get the final ranking using the whole video state and an LLM to rank the options
            ranking, confidence, completion = Ranking.derive_options_ranking_from_node_state(
                whole_video_summary=whole_video_summary,
                lexical_node_state_representation=unranked_node.state.get_lexical_representation(),
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                parse_confidence=self.parse_confidence
            )
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived ranking: {ranking}")
            logger.debug(f"Derived confidence: {confidence}")

            # update the ranking and ranking_confidence of the node
            unranked_node.state.ranking = ranking
            unranked_node.state.ranking_confidence = confidence if self.parse_confidence else None

        logger.info(f"Executed state rating operation: Ranking")

    @staticmethod
    def derive_options_ranking_from_node_state(
            whole_video_summary: str,
            lexical_node_state_representation: str,
            task: Task,
            api: API,
            prompt_template: str,
            completion_start: str,
            max_new_tokens: int,
            parse_confidence: bool = False
    ) -> (list[str], float, str):
        logger.info("Deriving options ranking from lexical node state representation using LLM...")

        prompt = prompt_template.format(
            whole_video_summary=whole_video_summary,
            lexical_node_state_representation=lexical_node_state_representation,
            question=task.question,
            option_0=task.options["option 0"],
            option_1=task.options["option 1"],
            option_2=task.options["option 2"],
            option_3=task.options["option 3"],
            option_4=task.options["option 4"]
        )

        logger.debug(f"Concatenated Prompt: {prompt}")
        logger.debug(f"Num chars of concatenated prompt: {len(prompt)}")
        logger.debug(f"Num words of concatenated prompt: {len(prompt.split())}")

        # get the final answer using the LLM
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            # TODO could be set to a specific configurable temperature
            temperature=None  # use the default temperature that is set for the model
        )
        logger.debug(f"Derived llm completion about final candidate: {completion}")

        # parse the ranking from the completion
        ranking = parse_list(completion)
        logger.debug(f"Parsed ranking list: {ranking}")

        ranking = [item.replace("_", " ") for item in ranking]
        ranking = [item.replace("{", "") for item in ranking]
        ranking = [item.replace("}", "") for item in ranking]
        logger.debug(f"Removed special characters from ranking list: {ranking}")

        # the default confidence will be zero if no float is found
        ranking_confidence = 0.0
        # if parse_confidence is True, parse the confidence score from the very first list element
        if parse_confidence:
            confidence = ranking[0]
            logger.debug(f"Extracted confidence item from ranking list: {confidence}")

            # regex pattern to match floats with two decimal places
            pattern = r"[-+]?\d*\.?\d+\b"

            # extract all matches of the pattern in the confidence string
            matches = re.findall(pattern, confidence)

            if len(matches) > 0:
                ranking_confidence = float(matches[0])
            logger.debug(f"Extracted confidence score from confidence item: {ranking_confidence}")

            ranking = ranking[1:]
            logger.debug(f"Removed confidence score from ranking list: {ranking}")

        option_ids = [re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item) for item in ranking]
        logger.debug(f"Extracted option ids from ranking list: {option_ids}")

        option_ids = [item[0] for item in option_ids if len(item) > 0]
        logger.debug(f"Removed empty option ids from ranking list: {option_ids}")

        option_ids = option_ids[:5]
        logger.debug(f"Trimmed option ids to 5: {option_ids}")

        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4"
        }

        # replace textual ids with the numeric ones
        for i in range(len(option_ids)):
            if option_ids[i] in word_to_number.keys():
                option_ids[i] = word_to_number[option_ids[i]]
        logger.debug(f"Replaced textual option ids with numeric ones: {option_ids}")

        number_to_option_id = {
            "0": "option 0",
            "1": "option 1",
            "2": "option 2",
            "3": "option 3",
            "4": "option 4"
        }

        # replace numbers with the option ids
        for i in range(len(option_ids)):
            if option_ids[i] in number_to_option_id.keys():
                option_ids[i] = number_to_option_id[option_ids[i]]
        logger.debug(f"Replaced numeric option ids with option ids: {option_ids}")

        # filter out illegal option ids
        option_ids = [option_id for option_id in option_ids if option_id in task.options.keys()]
        logger.debug(f"Filtered out illegal option ids from ranking list: {option_ids}")

        # randomly choose a ranking if the parsing failed (or failed partly)
        if len(option_ids) < 5:
            logger.warning("Failed to parse 5 option_ids from completion. Randomly fill the rest of the ranking.")

            # get the options that are not in the ranking
            remaining_option_ids = [option_id for option_id in task.options.keys() if option_id not in option_ids]

            # shuffle these remaining options
            random.shuffle(remaining_option_ids)

            # fill the rest of the ranking with the remaining options
            option_ids += remaining_option_ids
        logger.debug(f"Final ranking list: {option_ids}")

        # filter out duplicates while preserving the order
        option_ids = list(dict.fromkeys(option_ids))

        # make sure there are 5 elements in the ranking
        option_ids = option_ids[:5]

        # assert that the ranking is valid
        assert len(option_ids) == 5, f"Expected 5 option ids in the ranking, but got {len(option_ids)}."
        assert len(
            set(option_ids)) == 5, f"Expected 5 unique option ids in the ranking, but got {len(set(option_ids))}."
        assert all(option_id in task.options.keys() for option_id in
                   option_ids), f"Expected all option ids to be valid, but got {option_ids}."

        return option_ids, ranking_confidence, completion
#####################################################################################################

# Hi，you are an very experienced programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially for using Python completely to satisfy the requirement.
# Next please complete the code, including classes and functions for my algorithm framework under my instruction and decription

# help me to write a class named Ranking and it inherits Operation.  Its inputs include prompt: str, completion_start: str, max_new_tokens: int, parse_confidence: bool = False
# thne using these inputs to initialize the member variables: self.prompt,self.completion_start,self.max_new_tokens,self.parse_confidence
# the first function is named _execute and its inputs are  graph: Optional[Graph], api: Optional[API], target: Optional[Node] , returning none
# create variable named unranked_nodes from graph.get_unranked_nodes(). and get whole_video_summary from graph.root.state.spatial_node_state.action_captions_summary. Then from graph.root.state.task is task setted.
# iterate in unranked_nodes to get unranked_node 
# get the final ranking using the whole video state and an LLM to rank the options. Here need to call function from Ranking.derive_options_ranking_from_node_state, whose inputs includes whole_video_summary=whole_video_summary,lexical_node_state_representation=unranked_node.state.get_lexical_representation(),task=task,api=api,prompt_template=self.prompt,completion_start=self.completion_start,max_new_tokens=self.max_new_tokens,parse_confidence=self.parse_confidence. And then return variables ranking, confidence and completion
# update the ranking and ranking_confidence of the unranked _node with just gotten results, what is important that ranking_confidence updated only if self.parse_confidence is true
# the second function is named derive_options_ranking_from_node_state, it's a staticmethod. Its inputs includes whole_video_summary: str,lexical_node_state_representation: str,task: Task,pi: API,prompt_template: str, completion_start: str,max_new_tokens: int,parse_confidence: bool = False
# and then retun results (list[str], float, str). 
# using .format() to fill the waiting to be replaced place with variables in class. then logger the prompt and length of prompt and length of prompt.split()
# and then call class's function get_completion_from_text with inputs text=prompt,completion_start=completion_start, max_new_tokens=max_new_tokens,temperature=None ,return the result to completion
# next use parse_list to parse the ranking from completion and then iteratively pick item in rankings and replace("_", " ") ,replace("{", ""),replace("}", "")
# initalize the default as 0.0. if parse_confidence is true, set confidence to ranking [0]
# then regex pattern to match floats with two decimal places, using pattern = r"[-+]?\d*\.?\d+\b"
# extract all matches of the pattern in confidence string, using re.findall, and return to variable matches
# if length of matches is larger than 0, then set float(matches[0]) to ranking_confidence
# and then remove the first element in ranking list.
# next using re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item to iteratively handle each item in ranking and return result to option_ids
# then continuely iterate in options_ids to remove empty entry in option_ids
# next trimmed option ids to 5
# create a dictionary named word_to_number its keys are zero-four word str and correspongidng value is respective number string
# then iteratively to element in option_ids and replace the textual_ids with numeric ones in above dictionary word_to_number
# create a dictionary named number_to_option_id, whose keys are numeric ids and corresponding value are like "option 0" etc.
# then iteratively replace the numbers in option_ids with the option ids in number_to_option_id
# then iteratively remove the option_id in option_ids if the current element is not in task.options.keys()
# if the length of option_ids is less than 5 , this is seen as fail parsing 5 option_ids from completion 
# then iteratively get the options in task.options.keys that are not in the option_ids and save the results as list to remaining_option_ids
# then do random.shuffle for remaining_option_ids and fill the rest of option_ids with remaining_option_ids
# filter out duplicates in option_ids while preserving the order and then make sure there are 5 elements in the ranking
# finally using assert make sure the length of option_ids and all elements in option_ids are in option.task.keys()
# return option_ids, ranking_confidence, completion
class Ranking(Operation):
    def __init__(
        self,
        prompt: str,
        completion_start: str,
        max_new_tokens: int,
        parse_confidence: bool = False
    ):
        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.parse_confidence = parse_confidence

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node]
    ) -> None:
        if graph is None or api is None:
            logger.error("Graph or API is None.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        for unranked_node in unranked_nodes:
            ranking, confidence, completion = self.derive_options_ranking_from_node_state(
                whole_video_summary=whole_video_summary,
                lexical_node_state_representation=unranked_node.state.get_lexical_representation(),
                task=task,
                pi=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                parse_confidence=self.parse_confidence
            )

            unranked_node.ranking = ranking
            if self.parse_confidence:
                unranked_node.ranking_confidence = confidence

    @staticmethod
    def derive_options_ranking_from_node_state(
        whole_video_summary: str,
        lexical_node_state_representation: str,
        task: Task,
        pi: API,
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        parse_confidence: bool = False
    ) -> Tuple[List[str], float, str]:
        # Fill in the prompt template
        prompt = prompt_template.format(
            whole_video_summary=whole_video_summary,
            lexical_node_state_representation=lexical_node_state_representation,
            task=task
        )
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Prompt length: {len(prompt)}")
        logger.debug(f"Prompt split length: {len(prompt.split())}")

        # Call the API to get the completion
        completion = Ranking.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=None
        )

        # Parse the ranking from the completion
        rankings = Ranking.parse_list(completion)
        rankings = [item.replace("_", " ").replace("{", "").replace("}", "") for item in rankings]

        # Initialize confidence
        confidence = 0.0
        if parse_confidence and len(rankings) > 0:
            confidence_str = rankings[0]
            pattern = r"[-+]?\d*\.\d+\b"
            matches = re.findall(pattern, confidence_str)
            if matches:
                confidence = float(matches[0])
                rankings = rankings[1:]
            else:
                logger.warning("No confidence score found in completion.")
        else:
            rankings = rankings[:]

        # Extract option IDs using regex
        option_ids = []
        for item in rankings:
            matches = re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item.lower())
            option_ids.extend(matches)

        # Remove empty entries
        option_ids = [oid for oid in option_ids if oid]

        # Trim to 5 elements
        option_ids = option_ids[:5]

        # Map words to numbers
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4"
        }

        # Replace textual IDs with numeric ones
        option_ids = [word_to_number.get(oid, oid) for oid in option_ids]

        # Map numbers to option IDs
        number_to_option_id = {
            "0": "option 0",
            "1": "option 1",
            "2": "option 2",
            "3": "option 3",
            "4": "option 4"
        }

        # Replace numbers with option IDs
        option_ids = [number_to_option_id.get(oid, oid) for oid in option_ids]

        # Remove option IDs not in task.options
        option_ids = [oid for oid in option_ids if oid in task.options.keys()]

        # Check if we have less than 5 option IDs
        if len(option_ids) < 5:
            remaining_option_ids = [oid for oid in task.options.keys() if oid not in option_ids]
            random.shuffle(remaining_option_ids)
            option_ids.extend(remaining_option_ids[:5 - len(option_ids)])

        # Remove duplicates while preserving order
        seen = set()
        option_ids = [x for x in option_ids if not (x in seen or seen.add(x))]

        # Ensure there are exactly 5 elements
        option_ids = option_ids[:5]

        # Final assertions
        assert len(option_ids) == 5, "Failed to parse 5 option_ids from completion."
        assert all(oid in task.options.keys() for oid in option_ids), "Some option_ids are not in task.options."

        return option_ids, confidence, completion
    
#######################################################################################3
###################################################################################
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description.  
# please do not start writing, when i have not given you the partly code.
class Ranking(Operation):
    def __init__(self, prompt: str, completion_start: str, max_new_tokens: int, parse_confidence: bool = False):
        super().__init__()
        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.parse_confidence = parse_confidence

    def _execute(self, graph: Optional[Graph], api: Optional[API], target: Optional[Node]) -> None:
        if graph is None or api is None:
            logger.error("Graph or API is None.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        for unranked_node in unranked_nodes:
            logger.info(f"Rating state for unrated node {unranked_node}.")

            # Get the final ranking using the whole video state and an LLM to rank the options
            ranking, confidence, completion = self.derive_options_ranking_from_node_state(
                whole_video_summary=whole_video_summary,
                lexical_node_state_representation=unranked_node.state.get_lexical_representation(),
                task=task,
                pi=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                parse_confidence=self.parse_confidence
            )
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived ranking: {ranking}")
            logger.debug(f"Derived confidence: {confidence}")

            # Update the ranking and ranking_confidence of the node
            unranked_node.state.ranking = ranking
            if self.parse_confidence:
                unranked_node.state.ranking_confidence = confidence

        logger.info("Executed state rating operation: Ranking")

    @staticmethod
    def derive_options_ranking_from_node_state(
        whole_video_summary: str,
        lexical_node_state_representation: str,
        task: Task,
        pi: API,
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        parse_confidence: bool = False
    ) -> Tuple[List[str], float, str]:
        # Fill in the prompt template
        prompt = prompt_template.format(
            whole_video_summary=whole_video_summary,
            lexical_node_state_representation=lexical_node_state_representation,
            task=task
        )
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Prompt length: {len(prompt)}")
        logger.debug(f"Prompt split length: {len(prompt.split())}")

        # Call the API to get the completion
        completion = Ranking.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=None
        )

        # Parse the ranking from the completion
        rankings = Ranking.parse_list(completion)
        rankings = [item.replace("_", " ").replace("{", "").replace("}", "") for item in rankings]

        # Initialize confidence
        confidence = 0.0
        if parse_confidence and len(rankings) > 0:
            confidence_str = rankings[0]
            pattern = r"[-+]?\d*\.\d+\b"
            matches = re.findall(pattern, confidence_str)
            if matches:
                confidence = float(matches[0])
                rankings = rankings[1:]
            else:
                logger.warning("No confidence score found in completion.")
        else:
            rankings = rankings[:]

        # Extract option IDs using regex
        option_ids = []
        for item in rankings:
            matches = re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item.lower())
            option_ids.extend(matches)

        # Remove empty entries
        option_ids = [oid for oid in option_ids if oid]

        # Trim to 5 elements
        option_ids = option_ids[:5]

        # Map words to numbers
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4"
        }

        # Replace textual IDs with numeric ones
        option_ids = [word_to_number.get(oid, oid) for oid in option_ids]

        # Map numbers to option IDs
        number_to_option_id = {
            "0": "option 0",
            "1": "option 1",
            "2": "option 2",
            "3": "option 3",
            "4": "option 4"
        }

        # Replace numbers with option IDs
        option_ids = [number_to_option_id.get(oid, oid) for oid in option_ids]

        # Remove option IDs not in task.options
        option_ids = [oid for oid in option_ids if oid in task.options.keys()]

        # Check if we have less than 5 option IDs
        if len(option_ids) < 5:
            remaining_option_ids = [oid for oid in task.options.keys() if oid not in option_ids]
            random.shuffle(remaining_option_ids)
            option_ids.extend(remaining_option_ids[:5 - len(option_ids)])

        # Remove duplicates while preserving order
        seen = set()
        option_ids = [x for x in option_ids if not (x in seen or seen.add(x))]

        # Ensure there are exactly 5 elements
        option_ids = option_ids[:5]

        # Final assertions
        assert len(option_ids) == 5, f"Expected 5 option ids in the ranking, but got {len(option_ids)}."
        assert len(set(option_ids)) == 5, f"Expected 5 unique option ids in the ranking, but got {len(set(option_ids))}."
        assert all(option_id in task.options.keys() for option_id in option_ids), f"Expected all option ids to be valid, but got {option_ids}."

        return option_ids, confidence, completion
######################################################################################3
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.
######################################################################################33
###########################################################################################
    class Ranking(Operation):
    def __init__(
        self,
        prompt: str,
        completion_start: str,
        max_new_tokens: int,
        parse_confidence: bool = False
    ):
        super().__init__()
        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.parse_confidence = parse_confidence

    def _execute(
        self,
        graph: Optional[Graph],
        api: Optional[API],
        target: Optional[Node]
    ) -> None:
        if graph is None or api is None:
            logger.error("Graph or API is None.")
            return

        unranked_nodes = graph.get_unranked_nodes()
        whole_video_summary = graph.root.state.spatial_node_state.action_captions_summary
        task = graph.root.state.task

        for unranked_node in unranked_nodes:
            logger.info(f"Rating state for unrated node {unranked_node}.")

            # Get the final ranking using the whole video state and an LLM to rank the options
            ranking, confidence, completion = self.derive_options_ranking_from_node_state(
                whole_video_summary=whole_video_summary,
                lexical_node_state_representation=unranked_node.state.get_lexical_representation(),
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                parse_confidence=self.parse_confidence
            )
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived ranking: {ranking}")
            logger.debug(f"Derived confidence: {confidence}")

            # Update the ranking and ranking_confidence of the node
            unranked_node.state.ranking = ranking
            if self.parse_confidence:
                unranked_node.state.ranking_confidence = confidence

        logger.info("Executed state rating operation: Ranking")

    @staticmethod
    def derive_options_ranking_from_node_state(
        whole_video_summary: str,
        lexical_node_state_representation: str,
        task: Task,
        api: API,
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        parse_confidence: bool = False
    ) -> Tuple[List[str], float, str]:
        logger.info("Deriving options ranking from lexical node state representation using LLM...")

        # Fill in the prompt template
        prompt = prompt_template.format(
            whole_video_summary=whole_video_summary,
            lexical_node_state_representation=lexical_node_state_representation,
            question=task.question,
            option_0=task.options["option 0"],
            option_1=task.options["option 1"],
            option_2=task.options["option 2"],
            option_3=task.options["option 3"],
            option_4=task.options["option 4"]
        )

        logger.debug(f"Concatenated Prompt: {prompt}")
        logger.debug(f"Num chars of concatenated prompt: {len(prompt)}")
        logger.debug(f"Num words of concatenated prompt: {len(prompt.split())}")

        # Get the final answer using the LLM
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=None  # Use the default temperature that is set for the model
        )
        logger.debug(f"Derived LLM completion about final candidate: {completion}")

        # Parse the ranking from the completion
        ranking = Ranking.parse_list(completion)
        logger.debug(f"Parsed ranking list: {ranking}")

        # Remove special characters from ranking list
        ranking = [item.replace("_", " ") for item in ranking]
        ranking = [item.replace("{", "") for item in ranking]
        ranking = [item.replace("}", "") for item in ranking]
        logger.debug(f"Removed special characters from ranking list: {ranking}")

        # The default confidence will be zero if no float is found
        ranking_confidence = 0.0
        # If parse_confidence is True, parse the confidence score from the very first list element
        if parse_confidence and len(ranking) > 0:
            confidence = ranking[0]
            logger.debug(f"Extracted confidence item from ranking list: {confidence}")

            # Regex pattern to match floats
            pattern = r"[-+]?\d*\.\d+\b"

            # Extract all matches of the pattern in the confidence string
            matches = re.findall(pattern, confidence)

            if len(matches) > 0:
                ranking_confidence = float(matches[0])
            logger.debug(f"Extracted confidence score from confidence item: {ranking_confidence}")

            # Remove the first element from ranking as it was used for confidence
            ranking = ranking[1:]
            logger.debug(f"Removed confidence score from ranking list: {ranking}")

        # Extract option IDs using regex
        option_ids = []
        for item in ranking:
            matches = re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item.lower())
            option_ids.extend(matches)
        logger.debug(f"Extracted option ids from ranking list: {option_ids}")

        # Remove empty entries and keep the first match from each
        option_ids = [match[0] for match in [re.findall(r'(\d+|\b(?:zero|one|two|three|four)\b)', item.lower()) for item in ranking] if match]
        logger.debug(f"Removed empty option ids from ranking list: {option_ids}")

        # Trim to 5 elements
        option_ids = option_ids[:5]
        logger.debug(f"Trimmed option ids to 5: {option_ids}")

        # Map words to numbers
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4"
        }

        # Replace textual IDs with numeric ones
        option_ids = [word_to_number.get(oid, oid) for oid in option_ids]
        logger.debug(f"Replaced textual option ids with numeric ones: {option_ids}")

        # Map numbers to option IDs
        number_to_option_id = {
            "0": "option 0",
            "1": "option 1",
            "2": "option 2",
            "3": "option 3",
            "4": "option 4"
        }

        # Replace numbers with the option IDs
        option_ids = [number_to_option_id.get(oid, oid) for oid in option_ids]
        logger.debug(f"Replaced numeric option ids with option ids: {option_ids}")

        # Remove option IDs not in task.options
        option_ids = [oid for oid in option_ids if oid in task.options.keys()]
        logger.debug(f"Filtered out illegal option ids from ranking list: {option_ids}")

        # Check if we have less than 5 option IDs
        if len(option_ids) < 5:
            logger.warning("Failed to parse 5 option_ids from completion. Randomly filling the rest of the ranking.")

            # Get the options that are not in the ranking
            remaining_option_ids = [oid for oid in task.options.keys() if oid not in option_ids]

            # Shuffle these remaining options
            random.shuffle(remaining_option_ids)

            # Fill the rest of the ranking with the remaining options
            option_ids += remaining_option_ids[:5 - len(option_ids)]
        logger.debug(f"Final ranking list after filling: {option_ids}")

        # Remove duplicates while preserving order
        seen = set()
        option_ids = [x for x in option_ids if not (x in seen or seen.add(x))]
        logger.debug(f"Removed duplicates while preserving order: {option_ids}")

        # Ensure there are exactly 5 elements
        option_ids = option_ids[:5]
        logger.debug(f"Final option_ids trimmed to 5: {option_ids}")

        # Final assertions to ensure validity
        assert len(option_ids) == 5, f"Expected 5 option ids in the ranking, but got {len(option_ids)}."
        assert len(set(option_ids)) == 5, f"Expected 5 unique option ids in the ranking, but got {len(set(option_ids))}."
        assert all(option_id in task.options.keys() for option_id in option_ids), f"Expected all option ids to be valid, but got {option_ids}."

        return option_ids, ranking_confidence, completion