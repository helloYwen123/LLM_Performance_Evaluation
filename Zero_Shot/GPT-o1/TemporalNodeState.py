#Ground Truth
class TemporalNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_foreground: bool,
            use_relevance: bool,
            use_salience: bool,
            use_temporal_grounding_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.foreground_indicators: list[float] = []
        self.use_foreground = use_foreground

        self.relevance_indicators: list[int] = []
        self.use_relevance = use_relevance

        self.salience_indicators: list[int] = []
        self.use_salience = use_salience

        self.temporal_grounding_summary = None
        self.use_temporal_grounding_summary = use_temporal_grounding_summary

        if self.use_temporal_grounding_summary and not self.use_foreground and not self.use_relevance and not self.use_salience:
            logger.warning("Temporal grounding summary is enabled but no temporal "
                           "grounding is enabled. Enabling all temporal groundings.")
            self.use_foreground = True
            self.use_relevance = True
            self.use_salience = True

        logger.info("Initialized temporal node state")

    def get_lexical_representation(self) -> str:
        if not self.use_foreground and not self.use_relevance and not self.use_salience and not self.use_temporal_grounding_summary:
            return ""

        temporal_grounding_text = None
        if self.use_foreground or self.use_relevance or self.use_salience:
            logger.debug("Using temporal grounding in lexical representation.")
            temporal_grounding_text = self.get_textual_temporal_grounding()
            if temporal_grounding_text == "":
                temporal_grounding_text = None

        temporal_grounding_summary = None
        if self.use_temporal_grounding_summary:
            temporal_grounding_summary = self.temporal_grounding_summary

        # combine all information
        all_information = [temporal_grounding_text, temporal_grounding_summary]
        logger.debug(f"Collected all temporal information: {all_information}")

        # filter out None values
        all_information = [x for x in all_information if x is not None]
        logger.debug(f"Filtered out None values: {all_information}")

        if not all_information:
            return ""

        # add level indentation
        # choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            level_indentation = "                - "
            double_point = ":"
        elif self.lexical_representation == "sections" and self.use_temporal_grounding_summary:
            level_indentation = "\n"
            double_point = ""
        elif self.lexical_representation == "sections" and not self.use_temporal_grounding_summary:
            level_indentation = "\n###### "
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")
        all_information = [level_indentation + x for x in all_information]

        # add heading
        heading = "Temporal Information"
        heading = f"{heading}{double_point}"

        # add sub heading
        if self.use_temporal_grounding_summary:
            sub_heading = f"\n###### Temporal Grounding Summary"
        else:
            sub_heading = ""
        sub_heading = f"{sub_heading}{double_point}"
        all_information = [heading] + [sub_heading] + all_information
        logger.debug(f"Added heading: {all_information}")

        all_information = [x for x in all_information if x != ""]

        return "\n".join(all_information)

    def get_numeric_representation(self) -> torch.Tensor:
        raise NotImplementedError

    def get_json_representation(self) -> dict:
        return {
            "foreground_indicators": self.foreground_indicators,
            "foreground_ratio": self.get_foreground_ratio(),
            "relevance_indicators": self.relevance_indicators,
            "relevance_ratio": self.get_relevance_ratio(),
            "salience_indicators": self.salience_indicators,
            "salience_ratio": self.get_salience_ratio(),
            "temporal_grounding_text": self.get_lexical_representation(),
            "temporal_grounding_summary": self.temporal_grounding_summary,
            "use_foreground": self.use_foreground,
            "use_relevance": self.use_relevance,
            "use_salience": self.use_salience,
            "use_temporal_grounding_summary": self.use_temporal_grounding_summary
        }

    def get_textual_temporal_grounding(self) -> str:
        if not self.use_foreground and not self.use_relevance and not self.use_salience:
            return ""

        # choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = ": "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
        elif self.lexical_representation == "unformatted":
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = "Temporal Grounding of the Question within the Clip"
        foreground_ratio_text = (f"{round(self.get_foreground_ratio() * 100)}% of the frames within the clip are "
                                 f"foreground regarding the question.") if self.use_foreground else None
        relevance_ratio_text = (f"{round(self.get_relevance_ratio() * 100)}% of the frames within the clip are within "
                                f"the most relevant time interval regarding the question.") if self.use_relevance else None
        salience_ratio_text = (f"The mean salience of the question among all frames within the clip is "
                               f"{round(self.get_salience_ratio() * 100)}%.") if self.use_salience else None

        temporal_grounding = [foreground_ratio_text,
                              relevance_ratio_text,
                              salience_ratio_text]

        temporal_grounding = [x for x in temporal_grounding if x is not None]

        lexical_representation = "\n".join(temporal_grounding)

        if self.lexical_representation == "unformatted":
            return lexical_representation
        else:
            return f"{heading}{delimiter}{lexical_representation}"

    def get_foreground_ratio(self) -> float:
        # get ratio of all foreground indicators >= 0.5
        foreground_ratio = sum([1 for indicator in self.foreground_indicators if indicator >= 0.5]) / len(
            self.foreground_indicators) if len(self.foreground_indicators) > 0 else 0.0
        return foreground_ratio

    def get_relevance_ratio(self) -> float:
        relevance_ratio = sum(self.relevance_indicators) / len(self.relevance_indicators) if len(
            self.relevance_indicators) > 0 else 0.0
        return relevance_ratio

    def get_salience_ratio(self) -> float:
        salience_ratio = sum(self.salience_indicators) / len(self.salience_indicators) if len(
            self.salience_indicators) > 0 else 0.0
        return salience_ratio

    def __str__(self):
        return f"TemporalNodeState: {self.get_lexical_representation()}"
# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription


# please help me to finish a class named TemporalNodeState. It inherits BaseState. 
# It has several inputs: video_clip: VideoClip,task: Task,lexical_representation: str,use_foreground: bool,use_relevance: bool,use_salience: bool,use_temporal_grounding_summary: bool
# The initialization for father-class need video_clip and task.
# And it has some member variables waiting to be initializing: self.lexical_representation; self.foreground_indicators: list[float] = [];self.use_foreground;self.relevance_indicators: list[int] = [];self.use_relevance;self.salience_indicators: list[int] = [];self.use_salience;self.temporal_grounding_summary = None;self.use_temporal_grounding_summary
# if self.use_temporal_grounding_summary is true and self.use_foreground, self.use_relevance and self.use_salience are false, then logger warning: Temporal grounding summary is enabled but no temporal grounding is enabled. Enabling all temporal groundings.
# and next set self.use_foreground, self.use_relevance and self.use_salience to true.
# the first member function is get_lexical_representation returning string type
# if self.use_foreground, self.use_relevance, self.use_salience and self.use_temporal_grounding_summary are all false, then return  ""
# initialize temporal_grounding_text and temporal_grounding_summary as None
# if one of above three sign variables is true, then call self.get_textual_temporal_grounding() and assgin temporal_grounding_text
# if temporal_grounding_text == "" then temporal_grounding_text keeps None
# if use_temporal_grounding_summary true, then assign self.temporal_grounding_summary to temporal_grounding_summary
# creates all_information list including temporal_grounding_text and temporal_grounding_summary 
# then fillter out None values in all_information
# choose the corresponding assignment based on the lexical representation:
# if lexical representation is "list" then  level_indentation = " and double_point = ":"
# if self.lexical_representation == "sections" and self.use_temporal_grounding_summary then level_indentation = "\n" and double_point = ""
# self.lexical_representation == "sections" and not self.use_temporal_grounding_summary 
# then level_indentation = "\n###### " and double_point = ""
# if other types and situation then raise ValueError 
# add level_indentation to every element in all_information
# add heading using "Temporal Information {double_point}", and if use summary then add subheading f"\n###### Temporal Grounding Summary"
# then all_information list will be added these heading and subheading
# remove "" in all_information and using "\n" to join all_information
# the second function is get_json_representation, returning dictionary: "foreground_indicators": self.foreground_indicators,"foreground_ratio": self.get_foreground_ratio(),"relevance_indicators": self.relevance_indicators,"relevance_ratio": self.get_relevance_ratio(), "salience_indicators": self.salience_indicators,"salience_ratio": self.get_salience_ratio(),"temporal_grounding_text": self.get_lexical_representation(),"temporal_grounding_summary": self.temporal_grounding_summary,"use_foreground": self.use_foreground,"use_relevance": self.use_relevance,"use_salience": self.use_salience,"use_temporal_grounding_summary": self.use_temporal_grounding_summary
# the next function is named get_textual_temporal_grounding returning string
# if self.use_foreground, self.use_relevance, self.use_salience and self.use_temporal_grounding_summary are all false, then return  ""
# choose the delimiter based on the lexical representation:
# if lexical representation is "list" then delimiter = ": "
# if lexical representation is "sections" then delimiter = "\n\n"
# if lexical_representation == "unformatted" then delimiter = ""
# raise error if any other types
# set heading here "Temporal Grounding of the Question within the Clip" and generate foreground_ratio_text string from self.get_foreground_ratio() * 100 if self.use_foreground is valid
# same two times process to relevance_ratio_text and salience_ratio_text
# create a variable list named temporal_grounding to include foreground_ratio_text,relevance_ratio_text,salience_ratio_text
# remove the none element in temporal_grounding and use join "\n" temporal_grounding and assign it to lexical_representation
# if self.lexical_representation == "unformatted" directly return lexical_representation, otherwise return f"{heading}{delimiter}{lexical_representation}"
# the third function is named get_foreground_ratio, returning float variable
# it gets ratio of all foreground indicators >= 0.5 ,self.foreground_indicators is a list where each entry represents a foreground indicator for a frame. 
# if the list foreground_indicators is empty, get_foreground_ratio will return 0.0 
# If indicator >= 0.5, it counts the frame as foreground, adding 1 to the count; otherwise, it’s ignored. By diving the total number will it get the ratio
# similar process will be taken in next two functions respectively named get_relevance_ratio and get_salience_ratio
# But what is differrent if indicator >0 , it counts the frame as relevance or salience
# The last function is named __str__  it will call get_lexical_representation. 
class TemporalNodeState(BaseState):
    def __init__(
        self,
        video_clip: VideoClip,
        task: Task,
        lexical_representation: str,
        use_foreground: bool,
        use_relevance: bool,
        use_salience: bool,
        use_temporal_grounding_summary: bool
    ):
        super().__init__(video_clip, task)
        self.lexical_representation: str = lexical_representation
        self.foreground_indicators: List[float] = []
        self.use_foreground: bool = use_foreground
        self.relevance_indicators: List[int] = []
        self.use_relevance: bool = use_relevance
        self.salience_indicators: List[int] = []
        self.use_salience: bool = use_salience
        self.temporal_grounding_summary: Optional[str] = None
        self.use_temporal_grounding_summary: bool = use_temporal_grounding_summary

        # Check for temporal grounding summary without any grounding enabled
        if self.use_temporal_grounding_summary and not (
            self.use_foreground or self.use_relevance or self.use_salience
        ):
            logger.warning(
                "Temporal grounding summary is enabled but no temporal grounding is enabled. Enabling all temporal groundings."
            )
            self.use_foreground = True
            self.use_relevance = True
            self.use_salience = True

    def get_lexical_representation(self) -> str:
        if not (
            self.use_foreground
            or self.use_relevance
            or self.use_salience
            or self.use_temporal_grounding_summary
        ):
            return ""

        temporal_grounding_text: Optional[str] = None
        temporal_grounding_summary: Optional[str] = None

        if self.use_foreground or self.use_relevance or self.use_salience:
            temporal_grounding_text = self.get_textual_temporal_grounding()
            if temporal_grounding_text == "":
                temporal_grounding_text = None

        if self.use_temporal_grounding_summary:
            temporal_grounding_summary = self.temporal_grounding_summary

        all_information = [temporal_grounding_text, temporal_grounding_summary]
        all_information = [info for info in all_information if info is not None]

        # Determine indentation and delimiter based on lexical representation
        if self.lexical_representation == "list":
            level_indentation = " "
            double_point = ":"
        elif self.lexical_representation == "sections":
            if self.use_temporal_grounding_summary:
                level_indentation = "\n"
                double_point = ""
            else:
                level_indentation = "\n###### "
                double_point = ""
        else:
            raise ValueError(f"Unsupported lexical representation type: {self.lexical_representation}")

        # Add indentation to each element
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add headings
        heading = f"Temporal Information{double_point}"
        if self.use_temporal_grounding_summary:
            subheading = "\n###### Temporal Grounding Summary"
            all_information.insert(0, heading)
            all_information.insert(1, subheading)
        else:
            all_information.insert(0, heading)

        # Remove empty strings and join
        all_information = [info for info in all_information if info.strip() != ""]
        return "\n".join(all_information)

    def get_json_representation(self) -> Dict[str, object]:
        return {
            "foreground_indicators": self.foreground_indicators,
            "foreground_ratio": self.get_foreground_ratio(),
            "relevance_indicators": self.relevance_indicators,
            "relevance_ratio": self.get_relevance_ratio(),
            "salience_indicators": self.salience_indicators,
            "salience_ratio": self.get_salience_ratio(),
            "temporal_grounding_text": self.get_lexical_representation(),
            "temporal_grounding_summary": self.temporal_grounding_summary,
            "use_foreground": self.use_foreground,
            "use_relevance": self.use_relevance,
            "use_salience": self.use_salience,
            "use_temporal_grounding_summary": self.use_temporal_grounding_summary
        }

    def get_textual_temporal_grounding(self) -> str:
        if not (
            self.use_foreground
            or self.use_relevance
            or self.use_salience
            or self.use_temporal_grounding_summary
        ):
            return ""

        # Determine delimiter based on lexical representation
        if self.lexical_representation == "list":
            delimiter = ": "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
        elif self.lexical_representation == "unformatted":
            delimiter = ""
        else:
            raise ValueError(f"Unsupported lexical representation type: {self.lexical_representation}")

        heading = "Temporal Grounding of the Question within the Clip"

        temporal_grounding = []

        if self.use_foreground:
            foreground_ratio_text = f"Foreground Ratio: {self.get_foreground_ratio() * 100:.2f}%"
            temporal_grounding.append(foreground_ratio_text)
        if self.use_relevance:
            relevance_ratio_text = f"Relevance Ratio: {self.get_relevance_ratio() * 100:.2f}%"
            temporal_grounding.append(relevance_ratio_text)
        if self.use_salience:
            salience_ratio_text = f"Salience Ratio: {self.get_salience_ratio() * 100:.2f}%"
            temporal_grounding.append(salience_ratio_text)

        # Remove None elements and join
        temporal_grounding = [tg for tg in temporal_grounding if tg is not None]
        lexical_representation = delimiter.join(temporal_grounding)

        if self.lexical_representation == "unformatted":
            return lexical_representation
        else:
            return f"{heading}{delimiter}{lexical_representation}"

    def get_foreground_ratio(self) -> float:
        if not self.foreground_indicators:
            return 0.0
        count = sum(1 for indicator in self.foreground_indicators if indicator >= 0.5)
        ratio = count / len(self.foreground_indicators)
        return ratio

    def get_relevance_ratio(self) -> float:
        if not self.relevance_indicators:
            return 0.0
        count = sum(1 for indicator in self.relevance_indicators if indicator > 0)
        ratio = count / len(self.relevance_indicators)
        return ratio

    def get_salience_ratio(self) -> float:
        if not self.salience_indicators:
            return 0.0
        count = sum(1 for indicator in self.salience_indicators if indicator > 0)
        ratio = count / len(self.salience_indicators)
        return ratio

    def __str__(self) -> str:
        return self.get_lexical_representation()
#######################################################################################################

# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. 
# Do not start writing when i have not given you the context of code

# a class named TemporalNodeState. It inherits BaseState. 
# It has several inputs: video_clip: VideoClip,task: Task,lexical_representation: str,use_foreground: bool,use_relevance: bool,use_salience: bool,use_temporal_grounding_summary: bool
# The initialization for father-class need video_clip and task.
# And it has some member variables waiting to be initializing: self.lexical_representation; self.foreground_indicators: list[float] = [];self.use_foreground;self.relevance_indicators: list[int] = [];self.use_relevance;self.salience_indicators: list[int] = [];self.use_salience;self.temporal_grounding_summary = None;self.use_temporal_grounding_summary
# if self.use_temporal_grounding_summary is true and self.use_foreground, self.use_relevance and self.use_salience are false, then logger warning: Temporal grounding summary is enabled but no temporal grounding is enabled. Enabling all temporal groundings.
# and next set self.use_foreground, self.use_relevance and self.use_salience to true.
# the first member function is get_lexical_representation returning string type
# if self.use_foreground, self.use_relevance, self.use_salience and self.use_temporal_grounding_summary are all false, then return  ""
# initialize temporal_grounding_text and temporal_grounding_summary as None
# if one of above three sign variables is true, then call self.get_textual_temporal_grounding() and assgin temporal_grounding_text
# if temporal_grounding_text == "" then temporal_grounding_text keeps None
# if use_temporal_grounding_summary true, then assign self.temporal_grounding_summary to temporal_grounding_summary
# creates all_information list including temporal_grounding_text and temporal_grounding_summary 
# then fillter out None values in all_information
# choose the corresponding assignment based on the lexical representation:
# if lexical representation is "list" then  level_indentation = " and double_point = ":"
# if self.lexical_representation == "sections" and self.use_temporal_grounding_summary then level_indentation = "\n" and double_point = ""
# self.lexical_representation == "sections" and not self.use_temporal_grounding_summary 
# then level_indentation = "\n###### " and double_point = ""
# if other types and situation then raise ValueError 
# add level_indentation to every element in all_information
# add heading using "Temporal Information {double_point}", and if use summary then add subheading f"\n###### Temporal Grounding Summary"
# then all_information list will be added these heading and subheading
# remove "" in all_information and using "\n" to join all_information
# the second function is get_json_representation, returning dictionary: "foreground_indicators": self.foreground_indicators,"foreground_ratio": self.get_foreground_ratio(),"relevance_indicators": self.relevance_indicators,"relevance_ratio": self.get_relevance_ratio(), "salience_indicators": self.salience_indicators,"salience_ratio": self.get_salience_ratio(),"temporal_grounding_text": self.get_lexical_representation(),"temporal_grounding_summary": self.temporal_grounding_summary,"use_foreground": self.use_foreground,"use_relevance": self.use_relevance,"use_salience": self.use_salience,"use_temporal_grounding_summary": self.use_temporal_grounding_summary
# the next function is named get_textual_temporal_grounding returning string
# if self.use_foreground, self.use_relevance, self.use_salience and self.use_temporal_grounding_summary are all false, then return  ""
# choose the delimiter based on the lexical representation:
# if lexical representation is "list" then delimiter = ": "
# if lexical representation is "sections" then delimiter = "\n\n"
# if lexical_representation == "unformatted" then delimiter = ""
# raise error if any other types
# set heading here "Temporal Grounding of the Question within the Clip" and generate foreground_ratio_text string from self.get_foreground_ratio() * 100 if self.use_foreground is valid
# same two times process to relevance_ratio_text and salience_ratio_text
# create a variable list named temporal_grounding to include foreground_ratio_text,relevance_ratio_text,salience_ratio_text
# remove the none element in temporal_grounding and use join "\n" temporal_grounding and assign it to lexical_representation
# if self.lexical_representation == "unformatted" directly return lexical_representation, otherwise return f"{heading}{delimiter}{lexical_representation}"
# the third function is named get_foreground_ratio, returning float variable
# it gets ratio of all foreground indicators >= 0.5 ,self.foreground_indicators is a list where each entry represents a foreground indicator for a frame. 
# if the list foreground_indicators is empty, get_foreground_ratio will return 0.0 
# If indicator >= 0.5, it counts the frame as foreground, adding 1 to the count; otherwise, it’s ignored. By diving the total number will it get the ratio
# similar process will be taken in next two functions respectively named get_relevance_ratio and get_salience_ratio
# But what is differrent if indicator >0 , it counts the frame as relevance or salience
# The last function is named __str__  it will call get_lexical_representation. 

class TemporalNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_foreground: bool,
            use_relevance: bool,
            use_salience: bool,
            use_temporal_grounding_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.foreground_indicators: List[float] = []
        self.use_foreground = use_foreground

        self.relevance_indicators: List[int] = []
        self.use_relevance = use_relevance

        self.salience_indicators: List[int] = []
        self.use_salience = use_salience

        self.temporal_grounding_summary: Optional[str] = None
        self.use_temporal_grounding_summary = use_temporal_grounding_summary

        if self.use_temporal_grounding_summary and not (self.use_foreground or self.use_relevance or self.use_salience):
            logger.warning("Temporal grounding summary is enabled but no temporal grounding is enabled. Enabling all temporal groundings.")
            self.use_foreground = True
            self.use_relevance = True
            self.use_salience = True

        logger.info("Initialized TemporalNodeState")

    def get_lexical_representation(self) -> str:
        if not (self.use_foreground or self.use_relevance or self.use_salience or self.use_temporal_grounding_summary):
            return ""

        temporal_grounding_text: Optional[str] = None
        if self.use_foreground or self.use_relevance or self.use_salience:
            logger.debug("Using temporal grounding in lexical representation.")
            temporal_grounding_text = self.get_textual_temporal_grounding()
            if temporal_grounding_text == "":
                temporal_grounding_text = None

        temporal_grounding_summary: Optional[str] = None
        if self.use_temporal_grounding_summary:
            temporal_grounding_summary = self.temporal_grounding_summary

        all_information = [temporal_grounding_text, temporal_grounding_summary]
        all_information = [info for info in all_information if info is not None]

        # Determine indentation and double_point based on lexical representation
        if self.lexical_representation == "list":
            level_indentation = " "
            double_point = ":"
        elif self.lexical_representation == "sections" and self.use_temporal_grounding_summary:
            level_indentation = "\n"
            double_point = ""
        elif self.lexical_representation == "sections" and not self.use_temporal_grounding_summary:
            level_indentation = "\n###### "
            double_point = ""
        else:
            raise ValueError(f"Unsupported lexical representation type: {self.lexical_representation}")

        # Add indentation to each element in all_information
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add heading and subheading
        heading = f"Temporal Information{double_point}"
        if self.use_temporal_grounding_summary:
            subheading = "\n###### Temporal Grounding Summary"
            all_information.insert(0, heading)
            all_information.insert(1, subheading)
        else:
            all_information.insert(0, heading)

        logger.debug(f"Added heading and subheading: {all_information}")

        # Remove empty strings and join with newline
        all_information = [info for info in all_information if info.strip() != ""]
        return "\n".join(all_information)

    def get_json_representation(self) -> Dict[str, object]:
        return {
            "foreground_indicators": self.foreground_indicators,
            "foreground_ratio": self.get_foreground_ratio(),
            "relevance_indicators": self.relevance_indicators,
            "relevance_ratio": self.get_relevance_ratio(),
            "salience_indicators": self.salience_indicators,
            "salience_ratio": self.get_salience_ratio(),
            "temporal_grounding_text": self.get_lexical_representation(),
            "temporal_grounding_summary": self.temporal_grounding_summary,
            "use_foreground": self.use_foreground,
            "use_relevance": self.use_relevance,
            "use_salience": self.use_salience,
            "use_temporal_grounding_summary": self.use_temporal_grounding_summary
        }

    def get_textual_temporal_grounding(self) -> str:
        if not (self.use_foreground or self.use_relevance or self.use_salience or self.use_temporal_grounding_summary):
            return ""

        # Choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = ": "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
        elif self.lexical_representation == "unformatted":
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = "Temporal Grounding of the Question within the Clip"

        temporal_grounding = []

        if self.use_foreground:
            foreground_ratio = self.get_foreground_ratio() * 100
            foreground_ratio_text = f"{foreground_ratio:.2f}% of the frames within the clip are foreground regarding the question."
            temporal_grounding.append(foreground_ratio_text)

        if self.use_relevance:
            relevance_ratio = self.get_relevance_ratio() * 100
            relevance_ratio_text = f"{relevance_ratio:.2f}% of the frames within the clip are relevant to the question."
            temporal_grounding.append(relevance_ratio_text)

        if self.use_salience:
            salience_ratio = self.get_salience_ratio() * 100
            salience_ratio_text = f"{salience_ratio:.2f}% of the frames within the clip are salient regarding the question."
            temporal_grounding.append(salience_ratio_text)

        # Remove None elements (already handled by conditionals)
        lexical_representation = delimiter.join(temporal_grounding)

        if self.lexical_representation == "unformatted":
            return lexical_representation
        else:
            return f"{heading}{delimiter}{lexical_representation}"

    def get_foreground_ratio(self) -> float:
        """
        Calculates the ratio of foreground indicators that are >= 0.5.
        Returns 0.0 if there are no foreground indicators.
        """
        if not self.foreground_indicators:
            return 0.0
        count = sum(1 for indicator in self.foreground_indicators if indicator >= 0.5)
        ratio = count / len(self.foreground_indicators)
        logger.debug(f"Foreground ratio: {ratio:.2f}")
        return ratio

    def get_relevance_ratio(self) -> float:
        """
        Calculates the ratio of relevance indicators that are > 0.
        Returns 0.0 if there are no relevance indicators.
        """
        if not self.relevance_indicators:
            return 0.0
        count = sum(1 for indicator in self.relevance_indicators if indicator > 0)
        ratio = count / len(self.relevance_indicators)
        logger.debug(f"Relevance ratio: {ratio:.2f}")
        return ratio

    def get_salience_ratio(self) -> float:
        """
        Calculates the ratio of salience indicators that are > 0.
        Returns 0.0 if there are no salience indicators.
        """
        if not self.salience_indicators:
            return 0.0
        count = sum(1 for indicator in self.salience_indicators if indicator > 0)
        ratio = count / len(self.salience_indicators)
        logger.debug(f"Salience ratio: {ratio:.2f}")
        return ratio

    def __str__(self) -> str:
        return self.get_lexical_representation()
##############################################################################################################################3
##############################################################################################################################


class TemporalNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_foreground: bool,
            use_relevance: bool,
            use_salience: bool,  # Added missing parameter
            use_temporal_grounding_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation: str = lexical_representation

        self.foreground_indicators: List[float] = []
        self.use_foreground: bool = use_foreground

        self.relevance_indicators: List[int] = []
        self.use_relevance: bool = use_relevance

        self.salience_indicators: List[int] = []
        self.use_salience: bool = use_salience

        self.temporal_grounding_summary: Optional[str] = None
        self.use_temporal_grounding_summary: bool = use_temporal_grounding_summary

        if self.use_temporal_grounding_summary and not (self.use_foreground or self.use_relevance or self.use_salience):
            logger.warning(
                "Temporal grounding summary is enabled but no temporal grounding is enabled. "
                "Enabling all temporal groundings."
            )
            self.use_foreground = True
            self.use_relevance = True
            self.use_salience = True

        logger.info("Initialized TemporalNodeState")

    def get_lexical_representation(self) -> str:
        if not (self.use_foreground or self.use_relevance or self.use_salience or self.use_temporal_grounding_summary):
            logger.debug("All grounding flags are disabled. Returning empty string.")
            return ""

        temporal_grounding_text: Optional[str] = None
        if self.use_foreground or self.use_relevance or self.use_salience:
            logger.debug("Using temporal grounding in lexical representation.")
            temporal_grounding_text = self.get_textual_temporal_grounding()
            if temporal_grounding_text == "":
                temporal_grounding_text = None

        temporal_grounding_summary: Optional[str] = None
        if self.use_temporal_grounding_summary:
            temporal_grounding_summary = self.temporal_grounding_summary
            logger.debug("Temporal grounding summary is set.")

        # Combine all information
        all_information = [temporal_grounding_text, temporal_grounding_summary]
        logger.debug(f"Collected all temporal information: {all_information}")

        # Filter out None values
        all_information = [x for x in all_information if x is not None]
        logger.debug(f"Filtered out None values: {all_information}")

        if not all_information:
            return ""

        # Add level indentation
        # Choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            level_indentation = "    - "  # Adjusted indentation for list
            double_point = ":"
        elif self.lexical_representation == "sections" and self.use_temporal_grounding_summary:
            level_indentation = "\n"
            double_point = ""
        elif self.lexical_representation == "sections" and not self.use_temporal_grounding_summary:
            level_indentation = "\n###### "
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        # Add indentation to each element in all_information
        all_information = [f"{level_indentation}{x}" for x in all_information]
        logger.debug(f"All information after indentation: {all_information}")

        # Add heading
        heading = f"Temporal Information{double_point}"

        # Add sub heading
        if self.use_temporal_grounding_summary:
            sub_heading = f"\n###### Temporal Grounding Summary"
        else:
            sub_heading = ""
        sub_heading = f"{sub_heading}"
        
        # Combine heading, subheading, and all information
        if self.use_temporal_grounding_summary:
            all_information = [heading, sub_heading] + all_information
        else:
            all_information = [heading] + all_information

        logger.debug(f"Added heading: {all_information}")

        # Remove empty strings and join with newline
        all_information = [x for x in all_information if x.strip() != ""]
        final_representation = "\n".join(all_information)
        logger.debug(f"Final lexical representation: {final_representation}")

        return final_representation

    def get_numeric_representation(self) -> 'torch.Tensor':
        raise NotImplementedError("get_numeric_representation is not implemented yet.")

    def get_json_representation(self) -> Dict[str, object]:
        json_representation = {
            "foreground_indicators": self.foreground_indicators,
            "foreground_ratio": self.get_foreground_ratio(),
            "relevance_indicators": self.relevance_indicators,
            "relevance_ratio": self.get_relevance_ratio(),
            "salience_indicators": self.salience_indicators,
            "salience_ratio": self.get_salience_ratio(),
            "temporal_grounding_text": self.get_lexical_representation(),
            "temporal_grounding_summary": self.temporal_grounding_summary,
            "use_foreground": self.use_foreground,
            "use_relevance": self.use_relevance,
            "use_salience": self.use_salience,
            "use_temporal_grounding_summary": self.use_temporal_grounding_summary
        }
        logger.debug(f"JSON representation: {json_representation}")
        return json_representation

    def get_textual_temporal_grounding(self) -> str:
        if not (self.use_foreground or self.use_relevance or self.use_salience):
            logger.debug("No temporal grounding flags enabled. Returning empty string.")
            return ""

        # Choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = ": "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
        elif self.lexical_representation == "unformatted":
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = "Temporal Grounding of the Question within the Clip"
        temporal_grounding = [heading]

        if self.use_foreground:
            foreground_ratio = self.get_foreground_ratio() * 100
            foreground_ratio_text = f"{foreground_ratio:.2f}% of the frames within the clip are foreground regarding the question."
            temporal_grounding.append(foreground_ratio_text)
            logger.debug(f"Added foreground ratio text: {foreground_ratio_text}")

        if self.use_relevance:
            relevance_ratio = self.get_relevance_ratio() * 100
            relevance_ratio_text = f"{relevance_ratio:.2f}% of the frames within the clip are relevant to the question."
            temporal_grounding.append(relevance_ratio_text)
            logger.debug(f"Added relevance ratio text: {relevance_ratio_text}")

        if self.use_salience:
            salience_ratio = self.get_salience_ratio() * 100
            salience_ratio_text = f"{salience_ratio:.2f}% of the frames within the clip are salient regarding the question."
            temporal_grounding.append(salience_ratio_text)
            logger.debug(f"Added salience ratio text: {salience_ratio_text}")

        # Remove None elements (already handled by conditionals)
        # Exclude the heading for joining
        lexical_representation = delimiter.join(temporal_grounding[1:])
        logger.debug(f"Lexical representation before adding heading: {lexical_representation}")

        if self.lexical_representation == "unformatted":
            final_text = lexical_representation
        else:
            final_text = f"{heading}{delimiter}{lexical_representation}"

        logger.debug(f"Final textual temporal grounding: {final_text}")
        return final_text

    def get_foreground_ratio(self) -> float:
        """
        Calculates the ratio of foreground indicators that are >= 0.5.
        Returns 0.0 if there are no foreground indicators.
        """
        if not self.foreground_indicators:
            logger.debug("No foreground indicators available. Returning 0.0.")
            return 0.0
        count = sum(1 for indicator in self.foreground_indicators if indicator >= 0.5)
        ratio = count / len(self.foreground_indicators)
        logger.debug(f"Foreground ratio calculated: {ratio:.2f}")
        return ratio

    def get_relevance_ratio(self) -> float:
        """
        Calculates the ratio of relevance indicators that are > 0.
        Returns 0.0 if there are no relevance indicators.
        """
        if not self.relevance_indicators:
            logger.debug("No relevance indicators available. Returning 0.0.")
            return 0.0
        count = sum(1 for indicator in self.relevance_indicators if indicator > 0)
        ratio = count / len(self.relevance_indicators)
        logger.debug(f"Relevance ratio calculated: {ratio:.2f}")
        return ratio

    def get_salience_ratio(self) -> float:
        """
        Calculates the ratio of salience indicators that are > 0.
        Returns 0.0 if there are no salience indicators.
        """
        if not self.salience_indicators:
            logger.debug("No salience indicators available. Returning 0.0.")
            return 0.0
        count = sum(1 for indicator in self.salience_indicators if indicator > 0)
        ratio = count / len(self.salience_indicators)
        logger.debug(f"Salience ratio calculated: {ratio:.2f}")
        return ratio

    def __str__(self) -> str:
        """
        Returns the lexical representation of the temporal grounding.
        """
        return f"TemporalNodeState: {self.get_lexical_representation()}"