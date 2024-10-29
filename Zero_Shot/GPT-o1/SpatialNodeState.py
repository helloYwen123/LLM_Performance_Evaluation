#Ground Truth
class SpatialNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_object_detections: bool,
            use_action_captions_summary: bool,
            use_object_detections_summary: bool
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.action_captions: list[str] = []
        self.action_captions_summary: str = ""
        self.use_action_captions = use_action_captions
        self.use_action_captions_summary = use_action_captions_summary

        self.object_detections: list[list[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections = use_object_detections
        self.use_object_detections_summary = use_object_detections_summary

        logger.info("Initialized spatial node state")

    def get_lexical_representation(self) -> str:
        if not self.use_action_captions and not self.use_action_captions_summary and \
                not self.use_object_detections and not self.use_object_detections_summary:
            return ""

        # choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
            double_point = ""
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}"

        action_captions = None
        if self.use_action_captions:
            action_captions = [caption[:-1] if caption[-1] == "." else caption for caption in self.action_captions]
            action_captions = '. '.join(action_captions)
            if self.lexical_representation != "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}." if action_captions != "" else None

        action_captions_summary = None
        if self.use_action_captions_summary:
            # replace "video" with "clip"
            action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}" if action_captions_summary != "" else None

        object_detections = None
        if self.use_object_detections:
            # delimit the intervals by dots
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if self.lexical_representation != "unformatted":
                object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}." if unspecific_object_detections_text != "" else None

        object_detections_summary = None
        if self.use_object_detections_summary:
            # replace "video" with "clip"
            object_detections_summary = self.object_detections_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}" if object_detections_summary != "" else None

        # concatenate all information
        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]

        # filter out Nones and empty strings
        all_information = [info for info in all_information if info is not None]
        all_information = [info for info in all_information if info != ""]

        if not all_information:
            return ""

        # choose the level indentation based on the lexical representation
        if self.lexical_representation == "list":
            level_indentation = "                - "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        # add level indentation
        all_information = [level_indentation + info for info in all_information]

        # add heading
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)

        # join by linebreaks
        lexical_representation = "\n".join(all_information)

        return lexical_representation

    def get_numeric_representation(self) -> torch.Tensor:
        raise NotImplementedError

    def get_json_representation(self) -> dict:
        return {
            "action_captions": self.action_captions,
            "object_detections": self.object_detections,
            "action_captions_summary": self.action_captions_summary,
            "object_detections_summary": self.object_detections_summary,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_caption_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self):
        return f"SpatialNodeState: {self.get_lexical_representation()}"

    def get_textual_object_list(self) -> list[str]:
        # make first character of each object uppercase (looks better, have not ablated on that)
        unspecific_object_detections = [[obj[0].upper() + obj[1:] for obj in objects] for objects in
                                        self.object_detections]

        # delimit the objects of an interval by semicolons
        unspecific_object_detections_interval_texts = ['; '.join(objects) for objects in
                                                       unspecific_object_detections]

        # remove dots
        unspecific_object_detections_interval_texts = [text.replace(".", "") for text in
                                                       unspecific_object_detections_interval_texts]

        return unspecific_object_detections_interval_texts
###########################################################################3
# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# please help me to finish a class named SpatialNodeState it inherit from BaseState. It has several inputs:video_clip: VideoClip,task: Task,lexical_representation: str,use_action_captions: bool,use_object_detections: bool,use_action_captions_summary: bool,use_object_detections_summary: bool
# and its fatherclass initialization needs video_clip and task
# Its member variables are following：self.lexical_representation；self.action_captions；self.action_captions_summary；self.use_action_captions；self.use_action_captions_summary；self.object_detections；self.object_detections_summary；self.use_object_detections；self.use_object_detections_summary
# then report info after Initialization
# first function is get_lexical_representation, returning string type results
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then delimiter = " " ;double_point = ":"
# if is "sections" then delimiter = "\n\n";double_point = ""
# if is "unformatted" then delimiter = "";double_point = ""
# if other types then raise ValueError 
# create variable heading f"Spatial Information{double_point}"
# inititalize action_captions;action_captions_summary;object_detections;object_detections_summary as None
# if self.use_action_captions true, then remove the last letter of caption if the last one is '.'
# using '. 'join action_caption
# if lexical_representation is not "unformatted" , then go to if-branch. If action_caption is not "" then set action_captions to f"Action Captions{double_point}{delimiter}{action_captions}." Otherwise as None
# if self.use_action_captions_summary true, then replace "video" words in self.action_captions_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if action_captions_summary is not "" , then set action_captions_summary to f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}",Otherwise as None
# if self.use_object_detections true, use '. '  to join self.get_textual_object_list() , and assign it to variable unspecific_object_detections_text
# then  if lexical_representation is not "unformatted" and if unspecific_object_detections_text != "", then set object_detections to f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}.",Otherwise as None
# if self.use_object_detections_summary true,  then then replace "video" words in self.object_detections_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if object_detections_summary != "" , then set object_detections_summary to f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}", otherwise as None
# all_information variable is list including elements :action_captions,action_captions_summary,object_detections;object_detections_summary
# filter out Nones and empty strings in all_information
# if all_information not exists, then return "".
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then level_indentation = "
# if is "sections" then level_indentation = "\n###### "
# if is "unformatted" then level_indentation = ""
# if other types then raise ValueError 
# add corresponding level_indentation with each elements in all_information
# if self.lexical_representation != "unformatted" then add heading for all_information
# join all_information with "\n" then assign it to lexical_representation
# finally return lexical_representation
# the second function is named get_json_representation ,returning dictionary : "action_captions": self.action_captions,"object_detections": self.object_detections,"action_captions_summary": self.action_captions_summary, "object_detections_summary": self.object_detections_summary,"use_action_captions": self.use_action_captions,"use_object_detections": self.use_object_detections,"use_action_caption_summary": self.use_action_captions_summary,"use_object_detections_summary": self.use_object_detections_summary
# the third function __str__ will returns list[str] type variables. And calls function self.get_lexical_representation()
# the last function named get_textual_object_list will returns list[str] type variables.
# it will respectively pick objects from self.object_detections and then pick object from objects. Make the first letter big-writing and then join ";" between object in objects
# returns results named unspecific_object_detections_interval_texts

class SpatialNodeState(BaseState):
    def __init__(
        self,
        video_clip: 'VideoClip',
        task: 'Task',
        lexical_representation: str,
        use_action_captions: bool,
        use_object_detections: bool,
        use_action_captions_summary: bool,
        use_object_detections_summary: bool
    ):
        """
        Initializes the SpatialNodeState with the given parameters.

        Args:
            video_clip (VideoClip): The video clip associated with the state.
            task (Task): The task associated with the state.
            lexical_representation (str): The type of lexical representation.
            use_action_captions (bool): Flag to use action captions.
            use_object_detections (bool): Flag to use object detections.
            use_action_captions_summary (bool): Flag to use action captions summary.
            use_object_detections_summary (bool): Flag to use object detections summary.
        """
        super().__init__(video_clip, task)

        # Initialize member variables
        self.lexical_representation: str = lexical_representation
        self.use_action_captions: bool = use_action_captions
        self.use_object_detections: bool = use_object_detections
        self.use_action_captions_summary: bool = use_action_captions_summary
        self.use_object_detections_summary: bool = use_object_detections_summary

        self.lexical_representation_result: Optional[str] = None
        self.action_captions: Optional[str] = None
        self.action_captions_summary: Optional[str] = None
        self.object_detections: Optional[str] = None
        self.object_detections_summary: Optional[str] = None

        # Report initialization info
        self._report_initialization_info()

    def _report_initialization_info(self):
        """
        Reports the initialization information.
        """
        info = (
            f"SpatialNodeState initialized with:\n"
            f"  Lexical Representation: {self.lexical_representation}\n"
            f"  Use Action Captions: {self.use_action_captions}\n"
            f"  Use Action Captions Summary: {self.use_action_captions_summary}\n"
            f"  Use Object Detections: {self.use_object_detections}\n"
            f"  Use Object Detections Summary: {self.use_object_detections_summary}"
        )
        print(info)

    def get_lexical_representation(self) -> str:
        """
        Generates the lexical representation based on the initialized parameters.

        Returns:
            str: The formatted lexical representation.
        """
        # Determine delimiter and double_point based on lexical_representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
            level_indentation = " "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
            double_point = ""
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
            level_indentation = ""
        else:
            raise ValueError(f"Unsupported lexical_representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}"

        # Initialize member variables as None
        self.action_captions = None
        self.action_captions_summary = None
        self.object_detections = None
        self.object_detections_summary = None

        # Process Action Captions
        if self.use_action_captions:
            action_captions_list = self._get_action_captions()
            # Remove the last character if it's a '.'
            if action_captions_list and action_captions_list[-1].endswith('.'):
                action_captions_list[-1] = action_captions_list[-1][:-1]
            action_captions_joined = '. '.join(action_captions_list)
            if action_captions_joined:
                if self.lexical_representation != "unformatted":
                    self.action_captions = f"Action Captions{double_point}{delimiter}{action_captions_joined}."
                else:
                    self.action_captions = f"{action_captions_joined}."
            else:
                self.action_captions = None

        # Process Action Captions Summary
        if self.use_action_captions_summary:
            summary = self._get_action_captions_summary()
            summary = summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    self.action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{summary}"
                else:
                    self.action_captions_summary = summary
            else:
                self.action_captions_summary = None

        # Process Object Detections
        if self.use_object_detections:
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    self.object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    self.object_detections = f"{unspecific_object_detections_text}."
            else:
                self.object_detections = None

        # Process Object Detections Summary
        if self.use_object_detections_summary:
            summary = self._get_object_detections_summary()
            summary = summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    self.object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    self.object_detections_summary = summary
            else:
                self.object_detections_summary = None

        # Collect all information
        all_information = [
            self.action_captions,
            self.action_captions_summary,
            self.object_detections,
            self.object_detections_summary
        ]

        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]

        if not all_information:
            return ""

        # Add indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add heading if not unformatted
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)

        # Join all information with newline
        self.lexical_representation_result = "\n".join(all_information)
        return self.lexical_representation_result

    def get_json_representation(self) -> Dict[str, Optional[str]]:
        """
        Generates a JSON representation of the current state.

        Returns:
            dict: A dictionary containing the state information.
        """
        return {
            "action_captions": self.action_captions,
            "object_detections": self.object_detections,
            "action_captions_summary": self.action_captions_summary,
            "object_detections_summary": self.object_detections_summary,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_captions_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self) -> List[str]:
        """
        Returns a list containing the lexical representation.

        Returns:
            List[str]: A list with the lexical representation string.
        """
        return [self.get_lexical_representation()]

    def get_textual_object_list(self) -> List[str]:
        """
        Processes object detections to generate a list of textual descriptions.

        Returns:
            List[str]: A list of processed object detection strings.
        """
        unspecific_object_detections_interval_texts = []
        object_detections_data = self._get_object_detections()

        for objects in object_detections_data:
            # Capitalize the first letter of each object
            objects_capitalized = [obj.capitalize() for obj in objects]
            # Join objects with "; "
            joined_objects = "; ".join(objects_capitalized)
            unspecific_object_detections_interval_texts.append(joined_objects)

        return unspecific_object_detections_interval_texts
    
##################################################################################################33
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. 
# Do not start writing when i have not given you the context of code
# The class named SpatialNodeState it inherit from BaseState. It has several inputs:video_clip: VideoClip,task: Task,lexical_representation: str,use_action_captions: bool,use_object_detections: bool,use_action_captions_summary: bool,use_object_detections_summary: bool
# and its fatherclass initialization needs video_clip and task
# Its member variables are following：self.lexical_representation；self.action_captions；self.action_captions_summary；self.use_action_captions；self.use_action_captions_summary；self.object_detections；self.object_detections_summary；self.use_object_detections；self.use_object_detections_summary
# then report info after Initialization
# first function is get_lexical_representation, returning string type results
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then delimiter = " " ;double_point = ":"
# if is "sections" then delimiter = "\n\n";double_point = ""
# if is "unformatted" then delimiter = "";double_point = ""
# if other types then raise ValueError 
# create variable heading f"Spatial Information{double_point}"
# inititalize action_captions;action_captions_summary;object_detections;object_detections_summary as None
# if self.use_action_captions true, then remove the last letter of caption if the last one is '.'
# using '. 'join action_caption
# if lexical_representation is not "unformatted" , then go to if-branch. If action_caption is not "" then set action_captions to f"Action Captions{double_point}{delimiter}{action_captions}." Otherwise as None
# if self.use_action_captions_summary true, then replace "video" words in self.action_captions_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if action_captions_summary is not "" , then set action_captions_summary to f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}",Otherwise as None
# if self.use_object_detections true, use '. '  to join self.get_textual_object_list() , and assign it to variable unspecific_object_detections_text
# then  if lexical_representation is not "unformatted" and if unspecific_object_detections_text != "", then set object_detections to f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}.",Otherwise as None
# if self.use_object_detections_summary true,  then then replace "video" words in self.object_detections_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if object_detections_summary != "" , then set object_detections_summary to f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}", otherwise as None
# all_information variable is list including elements :action_captions,action_captions_summary,object_detections;object_detections_summary
# filter out Nones and empty strings in all_information
# if all_information not exists, then return "".
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then level_indentation = "
# if is "sections" then level_indentation = "\n###### "
# if is "unformatted" then level_indentation = ""
# if other types then raise ValueError 
# add corresponding level_indentation with each elements in all_information
# if self.lexical_representation != "unformatted" then add heading for all_information
# join all_information with "\n" then assign it to lexical_representation
# finally return lexical_representation
# the second function is named get_json_representation ,returning dictionary : "action_captions": self.action_captions,"object_detections": self.object_detections,"action_captions_summary": self.action_captions_summary, "object_detections_summary": self.object_detections_summary,"use_action_captions": self.use_action_captions,"use_object_detections": self.use_object_detections,"use_action_caption_summary": self.use_action_captions_summary,"use_object_detections_summary": self.use_object_detections_summary
# the third function __str__ will returns list[str] type variables. And calls function self.get_lexical_representation()
# the last function named get_textual_object_list will returns list[str] type variables.
# it will respectively pick objects from self.object_detections and then pick object from objects. Make the first letter big-writing and then join ";" between object in objects
# returns results named unspecific_object_detections_interval_texts
##########################################################################################

class SpatialNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_object_detections: bool,
            use_action_captions_summary: bool,
            use_object_detections_summary: bool
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation: str = lexical_representation

        self.action_captions: List[str] = []
        self.action_captions_summary: str = ""
        self.use_action_captions: bool = use_action_captions
        self.use_action_captions_summary: bool = use_action_captions_summary

        self.object_detections: List[List[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections: bool = use_object_detections
        self.use_object_detections_summary: bool = use_object_detections_summary

        logger.info("Initialized SpatialNodeState with the following parameters:")
        logger.info(f"  Lexical Representation: {self.lexical_representation}")
        logger.info(f"  Use Action Captions: {self.use_action_captions}")
        logger.info(f"  Use Action Captions Summary: {self.use_action_captions_summary}")
        logger.info(f"  Use Object Detections: {self.use_object_detections}")
        logger.info(f"  Use Object Detections Summary: {self.use_object_detections_summary}")

        # Initialize data (Assuming methods to fetch data are defined)
        self._initialize_data()

    def _initialize_data(self):
        """
        Initializes action captions, summaries, and object detections.
        Replace placeholder methods with actual data retrieval logic.
        """
        if self.use_action_captions:
            self.action_captions = self._fetch_action_captions()
            logger.info(f"Action Captions: {self.action_captions}")

        if self.use_action_captions_summary:
            self.action_captions_summary = self._fetch_action_captions_summary()
            logger.info(f"Action Captions Summary: {self.action_captions_summary}")

        if self.use_object_detections:
            self.object_detections = self._fetch_object_detections()
            logger.info(f"Object Detections: {self.object_detections}")

        if self.use_object_detections_summary:
            self.object_detections_summary = self._fetch_object_detections_summary()
            logger.info(f"Object Detections Summary: {self.object_detections_summary}")

    def get_lexical_representation(self) -> str:
        """
        Generates the lexical representation based on the initialized parameters.

        Returns:
            str: The formatted lexical representation.
        """
        if not (self.use_action_captions or self.use_action_captions_summary or
                self.use_object_detections or self.use_object_detections_summary):
            return ""

        # Determine delimiter and double_point based on lexical_representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
            level_indentation = "  - "  # Adjusted indentation for better readability
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
            double_point = ""
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
            level_indentation = ""
        else:
            raise ValueError(f"Unsupported lexical_representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}" if self.lexical_representation != "unformatted" else ""

        # Initialize member variables as None
        self.action_captions_str: Optional[str] = None
        self.action_captions_summary_str: Optional[str] = None
        self.object_detections_str: Optional[str] = None
        self.object_detections_summary_str: Optional[str] = None

        # Process Action Captions
        if self.use_action_captions and self.action_captions:
            # Remove the last character if it's a '.'
            if self.action_captions[-1].endswith('.'):
                self.action_captions[-1] = self.action_captions[-1][:-1]
            action_captions_joined = '. '.join(self.action_captions)
            if action_captions_joined:
                if self.lexical_representation != "unformatted":
                    self.action_captions_str = f"Action Captions{double_point}{delimiter}{action_captions_joined}."
                else:
                    self.action_captions_str = f"{action_captions_joined}."
            else:
                self.action_captions_str = None

        # Process Action Captions Summary
        if self.use_action_captions_summary and self.action_captions_summary:
            summary = self.action_captions_summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    self.action_captions_summary_str = f"Action Caption Summary{double_point}{delimiter}{summary}"
                else:
                    self.action_captions_summary_str = summary
            else:
                self.action_captions_summary_str = None

        # Process Object Detections
        if self.use_object_detections and self.object_detections:
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    self.object_detections_str = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    self.object_detections_str = f"{unspecific_object_detections_text}."
            else:
                self.object_detections_str = None

        # Process Object Detections Summary
        if self.use_object_detections_summary and self.object_detections_summary:
            summary = self.object_detections_summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    self.object_detections_summary_str = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    self.object_detections_summary_str = summary
            else:
                self.object_detections_summary_str = None

        # Collect all information
        all_information = [
            self.action_captions_str,
            self.action_captions_summary_str,
            self.object_detections_str,
            self.object_detections_summary_str
        ]

        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]

        if not all_information:
            return ""

        # Add level indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add heading if not unformatted
        if self.lexical_representation != "unformatted" and heading:
            all_information.insert(0, heading)

        # Join all information with newline
        lexical_representation_result = "\n".join(all_information)
        self.lexical_representation_result = lexical_representation_result

        return lexical_representation_result

    def get_json_representation(self) -> Dict[str, Optional[str]]:
        """
        Generates a JSON representation of the current state.

        Returns:
            dict: A dictionary containing the state information.
        """
        return {
            "action_captions": self.action_captions_str,
            "object_detections": self.object_detections_str,
            "action_captions_summary": self.action_captions_summary_str,
            "object_detections_summary": self.object_detections_summary_str,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_captions_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self) -> List[str]:
        """
        Returns a list containing the lexical representation.

        Returns:
            List[str]: A list with the lexical representation string.
        """
        return [self.get_lexical_representation()]

    def get_textual_object_list(self) -> List[str]:
        """
        Processes object detections to generate a list of textual descriptions.

        Returns:
            List[str]: A list of processed object detection strings.
        """
        unspecific_object_detections_interval_texts = []
        for objects in self.object_detections:
            # Capitalize the first letter of each object
            objects_capitalized = [obj.capitalize() for obj in objects]
            # Join objects with "; "
            joined_objects = "; ".join(objects_capitalized)
            unspecific_object_detections_interval_texts.append(joined_objects)
        return unspecific_object_detections_interval_texts
    ##########################################################################################################3
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start writing when i still given you my code.
# The class named SpatialNodeState it inherit from BaseState. It has several inputs:video_clip: VideoClip,task: Task,lexical_representation: str,use_action_captions: bool,use_object_detections: bool,use_action_captions_summary: bool,use_object_detections_summary: bool
# and its fatherclass initialization needs video_clip and task
# Its member variables are following：self.lexical_representation；self.action_captions；self.action_captions_summary；self.use_action_captions；self.use_action_captions_summary；self.object_detections；self.object_detections_summary；self.use_object_detections；self.use_object_detections_summary
# then report info after Initialization
# first function is get_lexical_representation, returning string type results
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then delimiter = " " ;double_point = ":"
# if is "sections" then delimiter = "\n\n";double_point = ""
# if is "unformatted" then delimiter = "";double_point = ""
# if other types then raise ValueError 
# create variable heading f"Spatial Information{double_point}"
# inititalize action_captions;action_captions_summary;object_detections;object_detections_summary as None
# if self.use_action_captions true, then remove the last letter of caption if the last one is '.'
# using '. 'join action_caption
# if lexical_representation is not "unformatted" , then go to if-branch. If action_caption is not "" then set action_captions to f"Action Captions{double_point}{delimiter}{action_captions}." Otherwise as None
# if self.use_action_captions_summary true, then replace "video" words in self.action_captions_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if action_captions_summary is not "" , then set action_captions_summary to f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}",Otherwise as None
# if self.use_object_detections true, use '. '  to join self.get_textual_object_list() , and assign it to variable unspecific_object_detections_text
# then  if lexical_representation is not "unformatted" and if unspecific_object_detections_text != "", then set object_detections to f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}.",Otherwise as None
# if self.use_object_detections_summary true,  then then replace "video" words in self.object_detections_summary as "clip"
# then  if lexical_representation is not "unformatted"  and if object_detections_summary != "" , then set object_detections_summary to f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}", otherwise as None
# all_information variable is list including elements :action_captions,action_captions_summary,object_detections;object_detections_summary
# filter out Nones and empty strings in all_information
# if all_information not exists, then return "".
# choose the corresponding assignment based on the lexical representation: 
# if is "list" then level_indentation = "
# if is "sections" then level_indentation = "\n###### "
# if is "unformatted" then level_indentation = ""
# if other types then raise ValueError 
# add corresponding level_indentation with each elements in all_information
# if self.lexical_representation != "unformatted" then add heading for all_information
# join all_information with "\n" then assign it to lexical_representation
# finally return lexical_representation
# the second function is named get_json_representation ,returning dictionary : "action_captions": self.action_captions,"object_detections": self.object_detections,"action_captions_summary": self.action_captions_summary, "object_detections_summary": self.object_detections_summary,"use_action_captions": self.use_action_captions,"use_object_detections": self.use_object_detections,"use_action_caption_summary": self.use_action_captions_summary,"use_object_detections_summary": self.use_object_detections_summary
# the third function __str__ will returns list[str] type variables. And calls function self.get_lexical_representation()
# the last function named get_textual_object_list will returns list[str] type variables.
# it will respectively pick objects from self.object_detections and then pick object from objects. Make the first letter big-writing and then join ";" between object in objects
# returns results named unspecific_object_detections_interval_texts
###############################################################################

class SpatialNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_object_detections: bool,
            use_action_captions_summary: bool,
            use_object_detections_summary: bool
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation: str = lexical_representation

        self.action_captions: List[str] = []
        self.action_captions_summary: str = ""
        self.use_action_captions: bool = use_action_captions
        self.use_action_captions_summary: bool = use_action_captions_summary

        self.object_detections: List[List[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections: bool = use_object_detections
        self.use_object_detections_summary: bool = use_object_detections_summary

        logger.info("Initialized SpatialNodeState with the following parameters:")
        logger.info(f"  Lexical Representation: {self.lexical_representation}")
        logger.info(f"  Use Action Captions: {self.use_action_captions}")
        logger.info(f"  Use Action Captions Summary: {self.use_action_captions_summary}")
        logger.info(f"  Use Object Detections: {self.use_object_detections}")
        logger.info(f"  Use Object Detections Summary: {self.use_object_detections_summary}")

        # Initialize data (Assuming methods to fetch data are defined)
        self._initialize_data()

    def _initialize_data(self):
        """
        Initializes action captions, summaries, and object detections.
        Replace placeholder methods with actual data retrieval logic.
        """
        if self.use_action_captions:
            self.action_captions = self._fetch_action_captions()
            logger.info(f"Action Captions: {self.action_captions}")

        if self.use_action_captions_summary:
            self.action_captions_summary = self._fetch_action_captions_summary()
            logger.info(f"Action Captions Summary: {self.action_captions_summary}")

        if self.use_object_detections:
            self.object_detections = self._fetch_object_detections()
            logger.info(f"Object Detections: {self.object_detections}")

        if self.use_object_detections_summary:
            self.object_detections_summary = self._fetch_object_detections_summary()
            logger.info(f"Object Detections Summary: {self.object_detections_summary}")

    def get_lexical_representation(self) -> str:
        """
        Generates the lexical representation based on the initialized parameters.

        Returns:
            str: The formatted lexical representation.
        """
        if not (self.use_action_captions or self.use_action_captions_summary or
                self.use_object_detections or self.use_object_detections_summary):
            return ""

        # Choose the delimiter and double_point based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
            level_indentation = "  - "  # Adjusted indentation for better readability
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
            double_point = ""
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}" if self.lexical_representation != "unformatted" else ""

        # Initialize member variables as None
        action_captions_str: Optional[str] = None
        action_captions_summary_str: Optional[str] = None
        object_detections_str: Optional[str] = None
        object_detections_summary_str: Optional[str] = None

        # Process Action Captions
        if self.use_action_captions and self.action_captions:
            # Remove the last character if it's a '.'
            processed_captions = [caption[:-1] if caption.endswith('.') else caption for caption in self.action_captions]
            action_captions_joined = '. '.join(processed_captions)
            if action_captions_joined:
                if self.lexical_representation != "unformatted":
                    action_captions_str = f"Action Captions{double_point}{delimiter}{action_captions_joined}."
                else:
                    action_captions_str = f"{action_captions_joined}."
            else:
                action_captions_str = None

        # Process Action Captions Summary
        if self.use_action_captions_summary and self.action_captions_summary:
            # Replace "video" with "clip"
            summary = self.action_captions_summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    action_captions_summary_str = f"Action Caption Summary{double_point}{delimiter}{summary}"
                else:
                    action_captions_summary_str = summary
            else:
                action_captions_summary_str = None

        # Process Object Detections
        if self.use_object_detections and self.object_detections:
            # Delimit the intervals by dots
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    object_detections_str = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    object_detections_str = f"{unspecific_object_detections_text}."
            else:
                object_detections_str = None

        # Process Object Detections Summary
        if self.use_object_detections_summary and self.object_detections_summary:
            # Replace "video" with "clip"
            summary = self.object_detections_summary.replace("video", "clip")
            if summary:
                if self.lexical_representation != "unformatted":
                    object_detections_summary_str = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    object_detections_summary_str = summary
            else:
                object_detections_summary_str = None

        # Concatenate all information
        all_information = [
            action_captions_str,
            action_captions_summary_str,
            object_detections_str,
            object_detections_summary_str
        ]

        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]

        if not all_information:
            return ""

        # Add level indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add heading if not unformatted
        if self.lexical_representation != "unformatted" and heading:
            all_information.insert(0, heading)

        # Join all information with newline
        lexical_representation_result = "\n".join(all_information)
        self.lexical_representation_result = lexical_representation_result

        return lexical_representation_result

    def get_numeric_representation(self) -> 'torch.Tensor':
        """
        Placeholder method for numeric representation.

        Raises:
            NotImplementedError: Indicates that this method should be implemented.

        Returns:
            torch.Tensor: Numeric representation (not implemented).
        """
        raise NotImplementedError("Numeric representation is not implemented.")

    def get_json_representation(self) -> Dict[str, Optional[str]]:
        """
        Generates a JSON representation of the current state.

        Returns:
            dict: A dictionary containing the state information.
        """
        return {
            "action_captions": self.action_captions,
            "object_detections": self.object_detections,
            "action_captions_summary": self.action_captions_summary,
            "object_detections_summary": self.object_detections_summary,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_captions_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the SpatialNodeState.

        Returns:
            str: The lexical representation of the state.
        """
        return f"SpatialNodeState: {self.get_lexical_representation()}"

    def get_textual_object_list(self) -> List[str]:
        """
        Processes object detections to generate a list of textual descriptions.

        Returns:
            List[str]: A list of processed object detection strings.
        """
        unspecific_object_detections_interval_texts = []
        for objects in self.object_detections:
            # Capitalize the first letter of each object
            objects_capitalized = [obj.capitalize() for obj in objects]
            # Join objects with "; "
            joined_objects = "; ".join(objects_capitalized)
            # Remove any trailing dots (if present)
            joined_objects = joined_objects.replace(".", "")
            unspecific_object_detections_interval_texts.append(joined_objects)
        return unspecific_object_detections_interval_texts