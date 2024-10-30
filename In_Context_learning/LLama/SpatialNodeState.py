class SpatialNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_object_detections: bool,
            use_action_captions_summary: bool,
            use_object_detections_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.action_captions = []
        self.action_captions_summary = None
        self.use_action_captions = use_action_captions
        self.use_action_captions_summary = use_action_captions_summary

        self.object_detections = []
        self.object_detections_summary = None
        self.use_object_detections = use_object_detections
        self.use_object_detections_summary = use_object_detections_summary

        logger.info("Initialized spatial node state")

    def get_lexical_representation(self) -> str:
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
        action_captions_summary = None
        object_detections = None
        object_detections_summary = None

        if self.use_action_captions:
            action_captions = '. '.join(self.action_captions).removesuffix('.')
            if self.lexical_representation!= "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}."
            else:
                action_captions = action_captions

        if self.use_action_captions_summary:
            self.action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation!= "unformatted" and self.action_captions_summary!= "":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{self.action_captions_summary}"

        if self.use_object_detections:
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if self.lexical_representation!= "unformatted" and unspecific_object_detections_text!= "":
                object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}"

        if self.use_object_detections_summary:
            self.object_detections_summary = self.object_detections_summary.replace("video", "clip")
            if self.lexical_representation!= "unformatted" and self.object_detections_summary!= "":
                object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{self.object_detections_summary}"

        all_information = [action_captions, action_captions_summary, object_detections, object_detections_summary]
        all_information = [x for x in all_information if x is not None and x!= ""]

        if not all_information:
            return ""

        if self.lexical_representation == "list":
            level_indentation = "                - "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        all_information = [level_indentation + x for x in all_information]

        if self.lexical_representation!= "unformatted":
            all_information = [heading] + all_information

        return "\n".join(all_information)

    def get_json_representation(self) -> dict:
        return {
            "action_captions": self.action_captions,
            "object_detections": self.object_detections,
            "action_captions_summary": self.action_captions_summary,
            "object_detections_summary": self.object_detections_summary,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_captions_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary,
        }

    def __str__(self):
        return self.get_lexical_representation()

    def get_textual_object_list(self) -> List[str]:
        unspecific_object_detections_interval_texts = []
        for objects in self.object_detections:
            for object in objects:
                unspecific_object_detections_interval_texts.append(object.capitalize())
        return unspecific_object_detections_interval_texts
##########################################################################################################################
 #Half Completion
###########################################################################################################################
# fail to understanding outputting Garbled text
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

        all_information = [x for x in all_information if x!= ""]

        return "\n".join(all_information)

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
 #################################################################################33#############
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
        super().__init__()

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
            if self.lexical_representation!= "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}." if action_captions!= "" else None

        action_captions_summary = None
        if self.use_action_captions_summary:
            # replace "video" with "clip"
            action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation!= "unformatted":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}" if action_captions_summary!= "" else None

        object_detections = None
        if self.use_object_detections:
            # delimit the intervals by dots
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if self.lexical_representation!= "unformatted":
                object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}." if unspecific_object_detections_text!= "" else None

        object_detections_summary = None
        if self.use_object_detections_summary:
            # replace "video" with "clip"
            object_detections_summary = self.object_detections_summary.replace("video", "clip")
            if self.lexical_representation!= "unformatted":
                object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}" if object_detections_summary!= "" else None

        # concatenate all information
        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]

        # filter out Nones and empty strings
        all_information = [info for info in all_information if info is not None]
        all_information = [info for info in all_information if info!= ""]

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
        if self.lexical_representation!= "unformatted":
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
            "use_action_captions_summary": self.use_action_captions_summary,
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