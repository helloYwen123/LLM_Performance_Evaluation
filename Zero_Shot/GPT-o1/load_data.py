###GPT-o1
#########################################################################################################
#######################################################################################################3#
#
#Ground_Truth


def load_data(answers_path: str, tasks_path: str, normalize: bool = False):
    # TODO implement a data preprocessing function (capital first letter and question mark / dot at the end)
    if "egoschema" in answers_path.lower():
        # load the answers (i.e. ground truths)
        answers = read_json_file(file_path=answers_path)
        logger.info(f"Loaded ground truth answers from {answers_path}.")

        # load the tasks (i.e. questions and options)
        tasks = read_json_file(file_path=tasks_path)
        logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

        data = []
        for video_id, answer_option_id in answers.items():
            # there is only one task per video in EgoSchema
            task = [task for task in tasks if task["q_uid"] == video_id][0]
            data.append({
                "video_id": video_id,
                "question": normalize_question(task["question"]) if normalize else task["question"],
                "options": {key: (normalize_answer(value) if normalize else value) for key, value in task.items() if
                            key.startswith("option")},
                "answer": f"option {answer_option_id}"
            })

        return data
    elif "nextqa" in answers_path.lower():
        # load all the data
        raw_data = read_csv_file(file_path=answers_path)
        logger.info(f"Loaded data from {answers_path}.")

        data = []
        for row in raw_data:

            # skip header row
            if row[0] == "video":
                continue

            # there can be multiple questions per video in NExT-QA
            data.append({
                "video_id": row[0],
                "question": normalize_question(row[4]) if normalize else row[4],
                "options": {
                    "option 0": normalize_answer(row[8]) if normalize else row[8],
                    "option 1": normalize_answer(row[9]) if normalize else row[9],
                    "option 2": normalize_answer(row[10]) if normalize else row[10],
                    "option 3": normalize_answer(row[11]) if normalize else row[11],
                    "option 4": normalize_answer(row[12]) if normalize else row[12]
                },
                "answer": f"option {row[5]}",
            })

        return data
    elif "intentqa" in answers_path.lower():
        # load all the data
        raw_data = read_csv_file(file_path=answers_path)
        logger.info(f"Loaded data from {answers_path}.")

        data = []
        for row in raw_data:

            # skip header row
            if row[0] == "video_id":
                continue

            # there can be multiple questions per video in IntentQA
            data.append({
                "video_id": row[0],
                "question": normalize_question(row[4]) if normalize else row[4],
                "options": {
                    "option 0": normalize_answer(row[8]) if normalize else row[8],
                    "option 1": normalize_answer(row[9]) if normalize else row[9],
                    "option 2": normalize_answer(row[10]) if normalize else row[10],
                    "option 3": normalize_answer(row[11]) if normalize else row[11],
                    "option 4": normalize_answer(row[12]) if normalize else row[12]
                },
                "answer": f"option {row[5]}",
            })

        return data
    elif "activitynet" in answers_path.lower():
        # load the answers (i.e. ground truths)
        answers = read_json_file(file_path=answers_path)
        logger.info(f"Loaded ground truth answers from {answers_path}.")

        # load the tasks (i.e. questions and options)
        tasks = read_json_file(file_path=tasks_path)
        logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

        data = []
        for item in answers:
            answer = item['answer']
            q_type = item['type']
            question_id = item['question_id']
            task = [task for task in tasks if task["question_id"] == question_id][0]
            question = normalize_question(task['question']) if normalize else task['question']
            video_name = question_id.rsplit('_', 1)[0]
            data.append({
                "video_id": video_name,
                "question": question,
                "options": {
                    "option 0": "N/A",
                    "option 1": "N/A",
                    "option 2": "N/A",
                    "option 3": "N/A",
                    "option 4": "N/A"
                },
                "answer": normalize_answer(answer) if normalize else answer
            })

        return data
    elif "videovista" in answers_path.lower():
        # load the answers (i.e. ground truths)
        raw_data = read_json_file(file_path=answers_path)
        logger.info(f"Loaded raw data including tasks and answers from {answers_path}.")

        data = []
        for entry in raw_data:
            data.append({
                # remove ".mp4" from the video name
                "video_id": entry["video_name"][:-4],
                "question": entry["Question"],
                "options": {
                    # note that VideoVISTA only has 4 options
                    "option 0": normalize_answer(entry["Answer_Choices"][0]) if normalize else entry["Answer_Choices"][0],
                    "option 1": normalize_answer(entry["Answer_Choices"][1]) if normalize else entry["Answer_Choices"][1],
                    "option 2": normalize_answer(entry["Answer_Choices"][2]) if normalize else entry["Answer_Choices"][2],
                    "option 3": normalize_answer(entry["Answer_Choices"][3]) if normalize else entry["Answer_Choices"][3]
                },
                "answer": f"option {entry['Answer']}"
            })

        return data
    else:
        err_msg = f"Dataset not supported: {answers_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    


    ###################################################################################################################################

# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# please help me to implement a function named load_data, its input parameters include answers_path: str, tasks_path: str, normalize: bool = False, and returns data variable
# first check via if-branch "egoschma" in answers_path.lower(), when existing call read_json_file to get result answers. and then logger.info
# run the same process for reading json from tasks_path
# iteratively get video_id and answer_option_id from answers.items()
# accept task from tasks if task["q_uid" is equal to video_id] , assign this task[0] to task variable
# do data.append() with dictionary "video_id" ,"question", "options","answer"
# here if normalize is true then call normalize_question, otherwise directly assign with task["question"]
# if key.startswith("option") is true then do iteration for task.items() , sequentially if normalize is true then do normalize_answer(value)
# regarding the input "answer" is assigned with "option{answer_option_id}"
# if "nextqa" in answers_path, firstly read_csv_file from answer_path to get raw_data
# Then iterate in raw_data. skip when ros[0] equals to "video". append data with dictionary
# dictionary includes "video_id": row[0];"question": normalize_question(row[4]) if normalize else row[4];
# "options": {"option 0": normalize_answer(row[8]) if normalize else row[8],
            #  "option 1": normalize_answer(row[9]) if normalize else row[9],
            #  "option 2": normalize_answer(row[10]) if normalize else row[10],
            #  "option 3": normalize_answer(row[11]) if normalize else row[11],
            #  "option 4": normalize_answer(row[12]) if normalize else row[12]
            # }
# "answer": f"option {row[5]}" 
# then return data
# if "intentqa" in answers_path.lower(), similarly do read_csv_file and logger.info, and run next same process as situation "nextqa"
# if "activitynet" in answers_path, do read_json_file for answer from answers_path and tasks from tasks_path, 
# iteratively get item from answers set question_id as item['question_id'], q_type = item['type'],answer = item['answer']
# accept task as variable from tasks if task["question_id"] == question_id][0]
# do the same checking process for choice of normalization in term of task['question']
# split question_id by '_' left and right side parts, and assign left part to video_name
# data.append using video_id";"question";"options": {
                    #"option 0": "N/A",
                    #"option 1": "N/A",
                    #"option 2": "N/A",
                    #"option 3": "N/A",
                    #"option 4": "N/A"
                #}
                # "answer": normalize_answer(answer) if normalize is true
# if "videovista" in answers_path , do read_json_file from answers_path to assgin raw_data
# iteratively get entry from raw_data
# append data list with "video_id": entry["video_name"][:-4],"question": entry["Question"],
# "options": {
                    # note that VideoVISTA only has 4 options
                #     "option 0": normalize_answer(entry["Answer_Choices"][0]) if normalize else entry["Answer_Choices"][0],
                #     "option 1": normalize_answer(entry["Answer_Choices"][1]) if normalize else entry["Answer_Choices"][1],
                #     "option 2": normalize_answer(entry["Answer_Choices"][2]) if normalize else entry["Answer_Choices"][2],
                #     "option 3": normalize_answer(entry["Answer_Choices"][3]) if normalize else entry["Answer_Choices"][3]
                # },
# "answer": f"option {entry['Answer']}" then return data
# finally to do logger error for no supported dataset

########################################################################################################3
#########################################################################################################
def load_data(
    answers_path: str,
    tasks_path: str,
    normalize: bool = False
) -> List[Dict[str, Any]]:
    """
    Loads and processes data based on the provided answers and tasks paths.
    
    Parameters:
    - answers_path (str): Path to the answers file.
    - tasks_path (str): Path to the tasks file.
    - normalize (bool): Whether to normalize questions and answers.
    
    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing processed data.
    """
    data = []
    answers_path_lower = answers_path.lower()

    if "egoschma" in answers_path_lower:
        # Handle "egoschma" dataset
        answers = read_json_file(answers_path)
        if answers is None:
            logger.error("Failed to load answers for 'egoschma' dataset.")
            return data
        logger.info("Loaded answers for 'egoschma' dataset.")

        tasks = read_json_file(tasks_path)
        if tasks is None:
            logger.error("Failed to load tasks for 'egoschma' dataset.")
            return data
        logger.info("Loaded tasks for 'egoschma' dataset.")

        for video_id, answer_option_id in answers.items():
            # Find the task with matching "q_uid"
            task = next((t for t in tasks if t.get("q_uid") == video_id), None)
            if not task:
                logger.warning(f"No task found for video_id: {video_id}")
                continue

            # Normalize question if required
            question = normalize_question(task["question"]) if normalize else task["question"]

            # Collect options
            options = {}
            for key, value in task.items():
                if key.startswith("option"):
                    options[key] = normalize_answer(value) if normalize else value

            # Assign answer
            answer = f"option {answer_option_id}"

            # Append to data
            data.append({
                "video_id": video_id,
                "question": question,
                "options": options,
                "answer": answer
            })

    elif "nextqa" in answers_path_lower:
        # Handle "nextqa" dataset
        raw_data = read_csv_file(answers_path)
        if not raw_data:
            logger.error("Failed to load answers for 'nextqa' dataset.")
            return data
        logger.info("Loaded answers for 'nextqa' dataset.")

        for row in raw_data:
            if not row or row[0].lower() == "video":
                continue  # Skip header or empty rows

            video_id = row[0]
            question = normalize_question(row[4]) if normalize else row[4]

            # Collect options from row[8] to row[12] -> "option 0" to "option 4"
            options = {}
            for i in range(5):  # 0 to 4
                option_key = f"option {i}"
                option_index = 8 + i
                if option_index < len(row):
                    option_value = row[option_index]
                    options[option_key] = normalize_answer(option_value) if normalize else option_value
                else:
                    options[option_key] = "N/A"  # Handle missing options

            # Assign answer
            answer = f"option {row[5]}" if len(row) > 5 else "option N/A"

            # Append to data
            data.append({
                "video_id": video_id,
                "question": question,
                "options": options,
                "answer": answer
            })

    elif "intentqa" in answers_path_lower:
        # Handle "intentqa" dataset (similar to "nextqa")
        raw_data = read_csv_file(answers_path)
        if not raw_data:
            logger.error("Failed to load answers for 'intentqa' dataset.")
            return data
        logger.info("Loaded answers for 'intentqa' dataset.")

        for row in raw_data:
            if not row or row[0].lower() == "video":
                continue  # Skip header or empty rows

            video_id = row[0]
            question = normalize_question(row[4]) if normalize else row[4]

            # Collect options from row[8] to row[12] -> "option 0" to "option 4"
            options = {}
            for i in range(5):  # 0 to 4
                option_key = f"option {i}"
                option_index = 8 + i
                if option_index < len(row):
                    option_value = row[option_index]
                    options[option_key] = normalize_answer(option_value) if normalize else option_value
                else:
                    options[option_key] = "N/A"  # Handle missing options

            # Assign answer
            answer = f"option {row[5]}" if len(row) > 5 else "option N/A"

            # Append to data
            data.append({
                "video_id": video_id,
                "question": question,
                "options": options,
                "answer": answer
            })

    elif "activitynet" in answers_path_lower:
        # Handle "activitynet" dataset
        answers = read_json_file(answers_path)
        if answers is None:
            logger.error("Failed to load answers for 'activitynet' dataset.")
            return data
        logger.info("Loaded answers for 'activitynet' dataset.")

        tasks = read_json_file(tasks_path)
        if tasks is None:
            logger.error("Failed to load tasks for 'activitynet' dataset.")
            return data
        logger.info("Loaded tasks for 'activitynet' dataset.")

        for item in answers:
            question_id = item.get('question_id')
            q_type = item.get('type')
            answer = item.get('answer')

            if not question_id:
                logger.warning("Missing 'question_id' in answer item.")
                continue

            # Find the task with matching "question_id"
            task = next((t for t in tasks if t.get("question_id") == question_id), None)
            if not task:
                logger.warning(f"No task found for question_id: {question_id}")
                continue

            # Normalize question if required
            question = normalize_question(task["question"]) if normalize else task["question"]

            # Split question_id by '_' and assign the left part to video_name
            video_name = question_id.split('_')[0] if '_' in question_id else question_id

            # Append to data with "N/A" options
            data.append({
                "video_id": video_name,
                "question": question,
                "options": {
                    "option 0": "N/A",
                    "option 1": "N/A",
                    "option 2": "N/A",
                    "option 3": "N/A",
                    "option 4": "N/A"
                },
                "answer": normalize_answer(answer) if normalize else answer
            })

    elif "videovista" in answers_path_lower:
        # Handle "videovista" dataset
        raw_data = read_json_file(answers_path)
        if raw_data is None:
            logger.error("Failed to load answers for 'videovista' dataset.")
            return data
        logger.info("Loaded answers for 'videovista' dataset.")

        for entry in raw_data:
            video_name = entry.get("video_name", "")
            if not video_name:
                logger.warning("Missing 'video_name' in entry.")
                continue
            video_id = video_name[:-4] if video_name.endswith('.mp4') else video_name

            question = entry.get("Question", "")
            if normalize:
                question = normalize_question(question)

            answer_choices = entry.get("Answer_Choices", [])
            if not isinstance(answer_choices, list):
                logger.warning("Invalid 'Answer_Choices' format.")
                answer_choices = []

            # Collect options (only 4 options)
            options = {}
            for i in range(4):
                option_key = f"option {i}"
                if i < len(answer_choices):
                    option_value = answer_choices[i]
                    options[option_key] = normalize_answer(option_value) if normalize else option_value
                else:
                    options[option_key] = "N/A"  # Handle missing options

            # Assign answer
            answer = entry.get("Answer")
            if answer is not None:
                answer_str = f"option {answer}"
            else:
                answer_str = "option N/A"

            # Append to data
            data.append({
                "video_id": video_id,
                "question": question,
                "options": options,
                "answer": answer_str
            })

    else:
        # Unsupported dataset
        logger.error("No supported dataset found in the answers_path.")
        return data

    logger.info(f"Data loading complete. Total records: {len(data)}")
    return data


####################################################################################################3
###############################################################################################################
##################################################################################################################
# Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. Please do not start when i still don't show you my code.
# please help me to implement a function named load_data, its input parameters include answers_path: str, tasks_path: str, normalize: bool = False, and returns data variable
# first check via if-branch "egoschma" in answers_path.lower(), when existing call read_json_file to get result answers. and then logger.info
# run the same process for reading json from tasks_path
# iteratively get video_id and answer_option_id from answers.items()
# accept task from tasks if task["q_uid" is equal to video_id] , assign this task[0] to task variable
# do data.append() with dictionary "video_id" ,"question", "options","answer"
# here if normalize is true then call normalize_question, otherwise directly assign with task["question"]
# if key.startswith("option") is true then do iteration for task.items() , sequentially if normalize is true then do normalize_answer(value)
# regarding the input "answer" is assigned with "option{answer_option_id}"
# if "nextqa" in answers_path, firstly read_csv_file from answer_path to get raw_data
# Then iterate in raw_data. skip when ros[0] equals to "video". append data with dictionary
# dictionary includes "video_id": row[0];"question": normalize_question(row[4]) if normalize else row[4];
# "options": {"option 0": normalize_answer(row[8]) if normalize else row[8],
            #  "option 1": normalize_answer(row[9]) if normalize else row[9],
            #  "option 2": normalize_answer(row[10]) if normalize else row[10],
            #  "option 3": normalize_answer(row[11]) if normalize else row[11],
            #  "option 4": normalize_answer(row[12]) if normalize else row[12]
            # }
# "answer": f"option {row[5]}" 
# then return data
# if "intentqa" in answers_path.lower(), similarly do read_csv_file and logger.info, and run next same process as situation "nextqa"
# if "activitynet" in answers_path, do read_json_file for answer from answers_path and tasks from tasks_path, 
# iteratively get item from answers set question_id as item['question_id'], q_type = item['type'],answer = item['answer']
# accept task as variable from tasks if task["question_id"] == question_id][0]
# do the same checking process for choice of normalization in term of task['question']
# split question_id by '_' left and right side parts, and assign left part to video_name
# data.append using video_id";"question";"options": {
                    #"option 0": "N/A",
                    #"option 1": "N/A",
                    #"option 2": "N/A",
                    #"option 3": "N/A",
                    #"option 4": "N/A"
                #}
                # "answer": normalize_answer(answer) if normalize is true
# if "videovista" in answers_path , do read_json_file from answers_path to assgin raw_data
# iteratively get entry from raw_data
# append data list with "video_id": entry["video_name"][:-4],"question": entry["Question"],
# "options": {
                    # note that VideoVISTA only has 4 options
                #     "option 0": normalize_answer(entry["Answer_Choices"][0]) if normalize else entry["Answer_Choices"][0],
                #     "option 1": normalize_answer(entry["Answer_Choices"][1]) if normalize else entry["Answer_Choices"][1],
                #     "option 2": normalize_answer(entry["Answer_Choices"][2]) if normalize else entry["Answer_Choices"][2],
                #     "option 3": normalize_answer(entry["Answer_Choices"][3]) if normalize else entry["Answer_Choices"][3]
                # },
# "answer": f"option {entry['Answer']}" then return data
# finally to do logger error for no supported dataset
############################################################################################################################################
##########################################################################################3

 def load_data(self, answers_path: str, tasks_path: str, normalize: bool = False) -> List[Dict[str, Any]]:
        """
        Loads and processes data based on the provided answers and tasks paths.

        Parameters:
        - answers_path (str): Path to the answers file.
        - tasks_path (str): Path to the tasks file.
        - normalize (bool): Whether to normalize questions and answers.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries containing processed data.
        """
        data = []
        answers_path_lower = answers_path.lower()

        if "egoschema" in answers_path_lower:
            # Handle "egoschema" dataset
            answers = self.read_json_file(file_path=answers_path)
            if answers is None:
                self.logger.error("Failed to load answers for 'egoschema' dataset.")
                return data
            self.logger.info("Loaded answers for 'egoschema' dataset.")

            tasks = self.read_json_file(file_path=tasks_path)
            if tasks is None:
                self.logger.error("Failed to load tasks for 'egoschema' dataset.")
                return data
            self.logger.info("Loaded tasks for 'egoschema' dataset.")

            for video_id, answer_option_id in answers.items():
                # There is only one task per video in EgoSchema
                try:
                    task = next(task for task in tasks if task.get("q_uid") == video_id)
                except StopIteration:
                    self.logger.warning(f"No task found for video_id: {video_id}")
                    continue

                # Normalize question if required
                question = self.normalize_question(task["question"]) if normalize else task["question"]

                # Collect options
                options = {
                    key: (self.normalize_answer(value) if normalize else value)
                    for key, value in task.items() if key.startswith("option")
                }

                # Assign answer
                answer = f"option {answer_option_id}"

                # Append to data
                data.append({
                    "video_id": video_id,
                    "question": question,
                    "options": options,
                    "answer": answer
                })

            return data

        elif "nextqa" in answers_path_lower or "intentqa" in answers_path_lower:
            # Handle "nextqa" and "intentqa" datasets
            raw_data = self.read_csv_file(file_path=answers_path)
            if not raw_data:
                dataset_name = "intentqa" if "intentqa" in answers_path_lower else "nextqa"
                self.logger.error(f"Failed to load answers for '{dataset_name}' dataset.")
                return data
            self.logger.info(f"Loaded data for '{'intentqa' if 'intentqa' in answers_path_lower else 'nextqa'}' dataset.")

            for row in raw_data:
                # Skip header row
                if not row or row[0].lower() == "video":
                    continue

                # Ensure the row has enough columns
                if len(row) < 13:  # For "nextqa" and "intentqa", expecting at least 13 columns
                    self.logger.warning(f"Row has insufficient columns: {row}")
                    continue

                # Assign fields
                video_id = row[0]
                question = self.normalize_question(row[4]) if normalize else row[4]
                options = {
                    "option 0": self.normalize_answer(row[8]) if normalize else row[8],
                    "option 1": self.normalize_answer(row[9]) if normalize else row[9],
                    "option 2": self.normalize_answer(row[10]) if normalize else row[10],
                    "option 3": self.normalize_answer(row[11]) if normalize else row[11],
                    "option 4": self.normalize_answer(row[12]) if normalize else row[12]
                }
                answer = f"option {row[5]}" if len(row) > 5 else "option N/A"

                # Append to data
                data.append({
                    "video_id": video_id,
                    "question": question,
                    "options": options,
                    "answer": answer
                })

            return data

        elif "activitynet" in answers_path_lower:
            # Handle "activitynet" dataset
            answers = self.read_json_file(file_path=answers_path)
            if answers is None:
                self.logger.error("Failed to load answers for 'activitynet' dataset.")
                return data
            self.logger.info("Loaded answers for 'activitynet' dataset.")

            tasks = self.read_json_file(file_path=tasks_path)
            if tasks is None:
                self.logger.error("Failed to load tasks for 'activitynet' dataset.")
                return data
            self.logger.info("Loaded tasks for 'activitynet' dataset.")

            for item in answers:
                question_id = item.get('question_id')
                q_type = item.get('type')
                answer = item.get('answer')

                if not question_id:
                    self.logger.warning("Missing 'question_id' in answer item.")
                    continue

                # Find the task with matching "question_id"
                try:
                    task = next(task for task in tasks if task.get("question_id") == question_id)
                except StopIteration:
                    self.logger.warning(f"No task found for question_id: {question_id}")
                    continue

                # Normalize question if required
                question = self.normalize_question(task['question']) if normalize else task['question']

                # Split question_id by '_' and assign the left part to video_name
                video_name = question_id.rsplit('_', 1)[0] if '_' in question_id else question_id

                # Append to data with "N/A" options
                data.append({
                    "video_id": video_name,
                    "question": question,
                    "options": {
                        "option 0": "N/A",
                        "option 1": "N/A",
                        "option 2": "N/A",
                        "option 3": "N/A",
                        "option 4": "N/A"
                    },
                    "answer": self.normalize_answer(answer) if normalize else answer
                })

            return data

        elif "videovista" in answers_path_lower:
            # Handle "videovista" dataset
            raw_data = self.read_json_file(file_path=answers_path)
            if raw_data is None:
                self.logger.error("Failed to load answers for 'videovista' dataset.")
                return data
            self.logger.info("Loaded answers for 'videovista' dataset.")

            for entry in raw_data:
                video_name = entry.get("video_name", "")
                if not video_name:
                    self.logger.warning("Missing 'video_name' in entry.")
                    continue
                video_id = video_name[:-4] if video_name.endswith('.mp4') else video_name

                question = entry.get("Question", "")
                if normalize:
                    question = self.normalize_question(question)

                answer_choices = entry.get("Answer_Choices", [])
                if not isinstance(answer_choices, list):
                    self.logger.warning("Invalid 'Answer_Choices' format.")
                    answer_choices = []

                # Collect options (only 4 options for VideoVISTA)
                options = {}
                for i in range(4):
                    option_key = f"option {i}"
                    if i < len(answer_choices):
                        option_value = answer_choices[i]
                        options[option_key] = self.normalize_answer(option_value) if normalize else option_value
                    else:
                        options[option_key] = "N/A"  # Handle missing options

                # Assign answer
                answer = entry.get("Answer")
                answer_str = f"option {answer}" if answer is not None else "option N/A"

                # Append to data
                data.append({
                    "video_id": video_id,
                    "question": question,
                    "options": options,
                    "answer": answer_str
                })

            return data

        else:
            # Unsupported dataset
            err_msg = f"Dataset not supported: {answers_path}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        
#############################################################################################################
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.
# the function named load_data, its input parameters include answers_path: str, tasks_path: str, normalize: bool = False, and returns data variable
# first check via if-branch "egoschma" in answers_path.lower(), when existing call read_json_file to get result answers. and then logger.info
# run the same process for reading json from tasks_path
# iteratively get video_id and answer_option_id from answers.items()
# accept task from tasks if task["q_uid" is equal to video_id] , assign this task[0] to task variable
# do data.append() with dictionary "video_id" ,"question", "options","answer"
# here if normalize is true then call normalize_question, otherwise directly assign with task["question"]
# if key.startswith("option") is true then do iteration for task.items() , sequentially if normalize is true then do normalize_answer(value)
# regarding the input "answer" is assigned with "option{answer_option_id}"
# if "nextqa" in answers_path, firstly read_csv_file from answer_path to get raw_data
# Then iterate in raw_data. skip when ros[0] equals to "video". append data with dictionary
# dictionary includes "video_id": row[0];"question": normalize_question(row[4]) if normalize else row[4];
# "options": {"option 0": normalize_answer(row[8]) if normalize else row[8],
            #  "option 1": normalize_answer(row[9]) if normalize else row[9],
            #  "option 2": normalize_answer(row[10]) if normalize else row[10],
            #  "option 3": normalize_answer(row[11]) if normalize else row[11],
            #  "option 4": normalize_answer(row[12]) if normalize else row[12]
            # }
# "answer": f"option {row[5]}" 
# then return data
# if "intentqa" in answers_path.lower(), similarly do read_csv_file and logger.info, and run next same process as situation "nextqa"
# if "activitynet" in answers_path, do read_json_file for answer from answers_path and tasks from tasks_path, 
# iteratively get item from answers set question_id as item['question_id'], q_type = item['type'],answer = item['answer']
# accept task as variable from tasks if task["question_id"] == question_id][0]
# do the same checking process for choice of normalization in term of task['question']
# split question_id by '_' left and right side parts, and assign left part to video_name
# data.append using video_id";"question";"options": {
                    #"option 0": "N/A",
                    #"option 1": "N/A",
                    #"option 2": "N/A",
                    #"option 3": "N/A",
                    #"option 4": "N/A"
                #}
                # "answer": normalize_answer(answer) if normalize is true
# if "videovista" in answers_path , do read_json_file from answers_path to assgin raw_data
# iteratively get entry from raw_data
# append data list with "video_id": entry["video_name"][:-4],"question": entry["Question"],
# "options": {
                    # note that VideoVISTA only has 4 options
                #     "option 0": normalize_answer(entry["Answer_Choices"][0]) if normalize else entry["Answer_Choices"][0],
                #     "option 1": normalize_answer(entry["Answer_Choices"][1]) if normalize else entry["Answer_Choices"][1],
                #     "option 2": normalize_answer(entry["Answer_Choices"][2]) if normalize else entry["Answer_Choices"][2],
                #     "option 3": normalize_answer(entry["Answer_Choices"][3]) if normalize else entry["Answer_Choices"][3]
                # },
# "answer": f"option {entry['Answer']}" then return data
# finally to do logger error for no supported dataset

######################################################################################################################################3
 def load_data(self, answers_path: str, tasks_path: str, normalize: bool = False) -> List[Dict[str, Any]]:
        """
        Loads and processes data based on the provided answers and tasks paths.

        Parameters:
        - answers_path (str): Path to the answers file.
        - tasks_path (str): Path to the tasks file.
        - normalize (bool): Whether to normalize questions and answers.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries containing processed data.
        """
        data = []
        answers_path_lower = answers_path.lower()

        if "egoschema" in answers_path_lower:
            # Handle "egoschema" dataset
            # Load the answers (i.e., ground truths)
            answers = self.read_json_file(file_path=answers_path)
            if answers is None:
                self.logger.error("Failed to load answers for 'egoschema' dataset.")
                return data
            self.logger.info(f"Loaded ground truth answers from {answers_path}.")

            # Load the tasks (i.e., questions and options)
            tasks = self.read_json_file(file_path=tasks_path)
            if tasks is None:
                self.logger.error("Failed to load tasks for 'egoschema' dataset.")
                return data
            self.logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

            data = []
            for video_id, answer_option_id in answers.items():
                # There is only one task per video in EgoSchema
                try:
                    task = next(task for task in tasks if task["q_uid"] == video_id)
                except StopIteration:
                    self.logger.warning(f"No task found for video_id: {video_id}")
                    continue

                data.append({
                    "video_id": video_id,
                    "question": self.normalize_question(task["question"]) if normalize else task["question"],
                    "options": {
                        key: self.normalize_answer(value) if normalize else value
                        for key, value in task.items() if key.startswith("option")
                    },
                    "answer": f"option {answer_option_id}"
                })

            return data

        elif "nextqa" in answers_path_lower:
            # Handle "nextqa" dataset
            # Load all the data
            raw_data = self.read_csv_file(file_path=answers_path)
            if not raw_data:
                self.logger.error(f"Failed to load data from {answers_path}.")
                return data
            self.logger.info(f"Loaded data from {answers_path}.")

            data = []
            for row in raw_data:
                # Skip header row
                if row[0].lower() == "video":
                    continue

                # Ensure the row has enough columns (expecting at least 13)
                if len(row) < 13:
                    self.logger.warning(f"Row has insufficient columns: {row}")
                    continue

                # Append data with dictionary
                data.append({
                    "video_id": row[0],
                    "question": self.normalize_question(row[4]) if normalize else row[4],
                    "options": {
                        "option 0": self.normalize_answer(row[8]) if normalize else row[8],
                        "option 1": self.normalize_answer(row[9]) if normalize else row[9],
                        "option 2": self.normalize_answer(row[10]) if normalize else row[10],
                        "option 3": self.normalize_answer(row[11]) if normalize else row[11],
                        "option 4": self.normalize_answer(row[12]) if normalize else row[12]
                    },
                    "answer": f"option {row[5]}"
                })

            return data

        elif "intentqa" in answers_path.lower():
            # Handle "intentqa" dataset
            # Load all the data
            raw_data = self.read_csv_file(file_path=answers_path)
            if not raw_data:
                self.logger.error(f"Failed to load data from {answers_path}.")
                return data
            self.logger.info(f"Loaded data from {answers_path}.")

            data = []
            for row in raw_data:
                # Skip header row
                if row[0].lower() == "video_id":
                    continue

                # Ensure the row has enough columns (expecting at least 13)
                if len(row) < 13:
                    self.logger.warning(f"Row has insufficient columns: {row}")
                    continue

                # Append data with dictionary
                data.append({
                    "video_id": row[0],
                    "question": self.normalize_question(row[4]) if normalize else row[4],
                    "options": {
                        "option 0": self.normalize_answer(row[8]) if normalize else row[8],
                        "option 1": self.normalize_answer(row[9]) if normalize else row[9],
                        "option 2": self.normalize_answer(row[10]) if normalize else row[10],
                        "option 3": self.normalize_answer(row[11]) if normalize else row[11],
                        "option 4": self.normalize_answer(row[12]) if normalize else row[12]
                    },
                    "answer": f"option {row[5]}"
                })

            return data

        elif "activitynet" in answers_path_lower:
            # Handle "activitynet" dataset
            # Load the answers (i.e., ground truths)
            answers = self.read_json_file(file_path=answers_path)
            if answers is None:
                self.logger.error("Failed to load answers for 'activitynet' dataset.")
                return data
            self.logger.info(f"Loaded ground truth answers from {answers_path}.")

            # Load the tasks (i.e., questions and options)
            tasks = self.read_json_file(file_path=tasks_path)
            if tasks is None:
                self.logger.error("Failed to load tasks for 'activitynet' dataset.")
                return data
            self.logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

            data = []
            for item in answers:
                answer = item.get('answer')
                q_type = item.get('type')
                question_id = item.get('question_id')

                if not question_id:
                    self.logger.warning("Missing 'question_id' in answer item.")
                    continue

                # Find the task with matching "question_id"
                try:
                    task = next(task for task in tasks if task["question_id"] == question_id)
                except StopIteration:
                    self.logger.warning(f"No task found for question_id: {question_id}")
                    continue

                # Normalize question if required
                question = self.normalize_question(task['question']) if normalize else task['question']

                # Split question_id by '_' and assign the left part to video_name
                video_name = question_id.rsplit('_', 1)[0]

                data.append({
                    "video_id": video_name,
                    "question": question,
                    "options": {
                        "option 0": "N/A",
                        "option 1": "N/A",
                        "option 2": "N/A",
                        "option 3": "N/A",
                        "option 4": "N/A"
                    },
                    "answer": self.normalize_answer(answer) if normalize else answer
                })

            return data

        elif "videovista" in answers_path_lower:
            # Handle "videovista" dataset
            # Load the answers (i.e., ground truths)
            raw_data = self.read_json_file(file_path=answers_path)
            if raw_data is None:
                self.logger.error("Failed to load answers for 'videovista' dataset.")
                return data
            self.logger.info(f"Loaded raw data including tasks and answers from {answers_path}.")

            data = []
            for entry in raw_data:
                # Remove ".mp4" from the video name
                video_name = entry.get("video_name", "")
                if not video_name:
                    self.logger.warning("Missing 'video_name' in entry.")
                    continue
                video_id = video_name[:-4] if video_name.endswith('.mp4') else video_name

                question = entry.get("Question", "")
                if normalize:
                    question = self.normalize_question(question)

                answer_choices = entry.get("Answer_Choices", [])
                if not isinstance(answer_choices, list):
                    self.logger.warning("Invalid 'Answer_Choices' format.")
                    answer_choices = []

                # Collect options (only 4 options for VideoVISTA)
                options = {}
                for i in range(4):
                    option_key = f"option {i}"
                    if i < len(answer_choices):
                        option_value = answer_choices[i]
                        options[option_key] = self.normalize_answer(option_value) if normalize else option_value
                    else:
                        options[option_key] = "N/A"  # Handle missing options

                # Assign answer
                answer = entry.get("Answer")
                answer_str = f"option {answer}" if answer is not None else "option N/A"

                # Append to data
                data.append({
                    "video_id": video_id,
                    "question": question,
                    "options": options,
                    "answer": answer_str
                })

            return data

        else:
            # Unsupported dataset
            err_msg = f"Dataset not supported: {answers_path}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

