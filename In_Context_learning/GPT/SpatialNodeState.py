class SpatialNodeState(BaseState):
    def __init__(
        self,
        video_clip: 'VideoClip',
        task: 'Task',
        lexical_representation: str,
        use_action_captions: bool,
        use_object_detections: bool,
        use_action_captions_summary: bool,
        use_object_detections_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)
        
        self.lexical_representation: str = lexical_representation
        
        self.action_captions: Optional[str] = None
        self.action_captions_summary: Optional[str] = None
        self.use_action_captions: bool = use_action_captions
        self.use_action_captions_summary: bool = use_action_captions_summary
        
        self.object_detections: Optional[str] = None
        self.object_detections_summary: Optional[str] = None
        self.use_object_detections: bool = use_object_detections
        self.use_object_detections_summary: bool = use_object_detections_summary
        
        logger.info("Initialized SpatialNodeState")

    def get_lexical_representation(self) -> str:
        # Determine delimiter and double_point based on lexical_representation
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
        
        # Initialize components
        action_captions = None
        action_captions_summary = None
        object_detections = None
        object_detections_summary = None
        
        # Process action_captions
        if self.use_action_captions and self.action_captions:
            captions = self.action_captions.rstrip('.')  # Remove trailing '.' if present
            action_caption_list = captions.split('. ')
            action_caption_list = [caption.strip() for caption in action_caption_list if caption.strip()]
            action_captions_joined = '. '.join(action_caption_list)
            if action_caption_list:
                if self.lexical_representation != "unformatted":
                    action_captions = f"Action Captions{double_point}{delimiter}{action_captions_joined}."
                else:
                    action_captions = f"{action_captions_joined}."
        
        # Process action_captions_summary
        if self.use_action_captions_summary and self.action_captions_summary:
            summary = self.action_captions_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    action_captions_summary = f"Action Captions Summary{double_point}{delimiter}{summary}"
                else:
                    action_captions_summary = summary
        
        # Process object_detections
        if self.use_object_detections and self.object_detections:
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    object_detections = f"{unspecific_object_detections_text}."
        
        # Process object_detections_summary
        if self.use_object_detections_summary and self.object_detections_summary:
            summary = self.object_detections_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    object_detections_summary = summary
        
        # Collect all information
        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]
        
        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]
        
        if not all_information:
            return ""
        
        # Determine level indentation
        if self.lexical_representation == "list":
            level_indentation = "- "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")
        
        # Apply indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]
        
        # Add heading if not unformatted
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)
        
        # Join all information
        lexical_representation = "\n".join(all_information)
        
        return lexical_representation

    def get_json_representation(self) -> Dict[str, Optional[str]]:
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

    def __str__(self) -> str:
        return self.get_lexical_representation()

    def get_textual_object_list(self) -> List[str]:
        unspecific_object_detections_interval_texts = []
        if not self.object_detections:
            return unspecific_object_detections_interval_texts
        
        for objects in self.object_detections:
            if not isinstance(objects, list):
                logger.warning("Each item in object_detections should be a list of objects.")
                continue
            formatted_objects = [obj.capitalize() for obj in objects if isinstance(obj, str) and obj.strip()]
            if formatted_objects:
                joined_objects = "; ".join(formatted_objects)
                unspecific_object_detections_interval_texts.append(joined_objects)
        
        return unspecific_object_detections_interval_texts
    ##########################################################################################################################
    #Half Completion
    ###########################################################################################################################
    class SpatialNodeState(BaseState):
    def __init__(
        self,
        video_clip: 'VideoClip',
        task: 'Task',
        lexical_representation: str,
        use_action_captions: bool,
        use_object_detections: bool,
        use_action_captions_summary: bool,
        use_object_detections_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)
        
        self.lexical_representation: str = lexical_representation
        
        self.action_captions: Optional[str] = None
        self.use_action_captions: bool = use_action_captions
        
        self.action_captions_summary: Optional[str] = None
        self.use_action_captions_summary: bool = use_action_captions_summary
        
        self.object_detections: Optional[List[List[str]]] = None
        self.use_object_detections: bool = use_object_detections
        
        self.object_detections_summary: Optional[str] = None
        self.use_object_detections_summary: bool = use_object_detections_summary
        
        if self.use_action_captions_summary and not self.use_action_captions:
            logger.warning("Action captions summary is enabled but action captions are not enabled. Enabling action captions.")
            self.use_action_captions = True
        
        if self.use_object_detections_summary and not self.use_object_detections:
            logger.warning("Object detections summary is enabled but object detections are not enabled. Enabling object detections.")
            self.use_object_detections = True
        
        logger.info("Initialized SpatialNodeState")
    
    def get_lexical_representation(self) -> str:
        # Determine delimiter and double_point based on lexical_representation
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
        
        # Initialize components
        action_captions = None
        action_captions_summary = None
        object_detections = None
        object_detections_summary = None
        
        # Process action_captions
        if self.use_action_captions and self.action_captions:
            captions = self.action_captions.rstrip('.')  # Remove trailing '.' if present
            action_caption_list = captions.split('. ')
            action_caption_list = [caption.strip() for caption in action_caption_list if caption.strip()]
            action_captions_joined = '. '.join(action_caption_list)
            if action_caption_list:
                if self.lexical_representation != "unformatted":
                    action_captions = f"Action Captions{double_point}{delimiter}{action_captions_joined}."
                else:
                    action_captions = f"{action_captions_joined}."
        
        # Process action_captions_summary
        if self.use_action_captions_summary and self.action_captions_summary:
            summary = self.action_captions_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{summary}"
                else:
                    action_captions_summary = summary
        
        # Process object_detections
        if self.use_object_detections and self.object_detections:
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    object_detections = f"{unspecific_object_detections_text}."
        
        # Process object_detections_summary
        if self.use_object_detections_summary and self.object_detections_summary:
            summary = self.object_detections_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    object_detections_summary = summary
        
        # Collect all information
        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]
        
        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]
        
        if not all_information:
            return ""
        
        # Determine level indentation
        if self.lexical_representation == "list":
            level_indentation = "- "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")
        
        # Apply indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]
        
        # Add heading if not unformatted
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)
        
        # Join all information
        lexical_representation = "\n".join(all_information)
        
        return lexical_representation
    
    def get_json_representation(self) -> Dict[str, Optional[str]]:
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
    
    def __str__(self) -> str:
        return self.get_lexical_representation()
    
    def get_textual_object_list(self) -> List[str]:
        unspecific_object_detections_interval_texts = []
        if not self.object_detections:
            return unspecific_object_detections_interval_texts
        
        for objects in self.object_detections:
            if not isinstance(objects, list):
                logger.warning("Each item in object_detections should be a list of objects.")
                continue
            formatted_objects = [obj.capitalize() for obj in objects if isinstance(obj, str) and obj.strip()]
            if formatted_objects:
                joined_objects = "; ".join(formatted_objects)
                unspecific_object_detections_interval_texts.append(joined_objects)
        
        return unspecific_object_detections_interval_texts
    
    ############################################################################################
    ############################################################################################
    # blank space 
    ############################################################################################
    class SpatialNodeState(BaseState):
    def __init__(
        self,
        video_clip: 'VideoClip',
        task: 'Task',
        lexical_representation: str,
        use_action_captions: bool,
        use_object_detections: bool,
        use_action_captions_summary: bool,
        use_object_detections_summary: bool,
    ):
        """
        Initializes the SpatialNodeState with the provided parameters.

        Args:
            video_clip (VideoClip): The video clip to analyze.
            task (Task): The task associated with the video clip.
            lexical_representation (str): The type of lexical representation ("list", "sections", "unformatted").
            use_action_captions (bool): Flag to use action captions.
            use_object_detections (bool): Flag to use object detections.
            use_action_captions_summary (bool): Flag to use action captions summary.
            use_object_detections_summary (bool): Flag to use object detections summary.
        """
        super().__init__(video_clip=video_clip, task=task)
        
        # Initialize member variables
        self.lexical_representation: str = lexical_representation
        
        self.action_captions: List[str] = []
        self.action_captions_summary: str = ""
        self.use_action_captions: bool = use_action_captions
        self.use_action_captions_summary: bool = use_action_captions_summary
        
        self.object_detections: List[List[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections: bool = use_object_detections
        self.use_object_detections_summary: bool = use_object_detections_summary
        
        # Handle dependencies between summaries and their main features
        if self.use_action_captions_summary and not self.use_action_captions:
            logger.warning("Action captions summary is enabled but action captions are not enabled. Enabling action captions.")
            self.use_action_captions = True
        
        if self.use_object_detections_summary and not self.use_object_detections:
            logger.warning("Object detections summary is enabled but object detections are not enabled. Enabling object detections.")
            self.use_object_detections = True
        
        logger.info("Initialized SpatialNodeState")

    def get_lexical_representation(self) -> str:
        """
        Generates the lexical representation of the spatial information.

        Returns:
            str: The formatted lexical representation.
        """
        if not self.use_action_captions and not self.use_action_captions_summary and \
           not self.use_object_detections and not self.use_object_detections_summary:
            return ""

        # Choose the delimiter and double_point based on lexical_representation
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

        # Initialize components
        action_captions_formatted = None
        action_captions_summary_formatted = None
        object_detections_formatted = None
        object_detections_summary_formatted = None

        # Process action_captions
        if self.use_action_captions and self.action_captions:
            # Remove trailing '.' from each caption if present
            processed_captions = [caption.rstrip('.') for caption in self.action_captions]
            # Join captions with '. '
            joined_captions = '. '.join(processed_captions)
            # Add trailing '.' if not empty
            if joined_captions:
                joined_captions += '.'
                if self.lexical_representation != "unformatted":
                    action_captions_formatted = f"Action Captions{double_point}{delimiter}{joined_captions}"
                else:
                    action_captions_formatted = joined_captions

        # Process action_captions_summary
        if self.use_action_captions_summary and self.action_captions_summary:
            # Replace "video" with "clip"
            summary = self.action_captions_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    action_captions_summary_formatted = f"Action Caption Summary{double_point}{delimiter}{summary}"
                else:
                    action_captions_summary_formatted = summary

        # Process object_detections
        if self.use_object_detections and self.object_detections:
            # Join textual object list with '. '
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if unspecific_object_detections_text:
                if self.lexical_representation != "unformatted":
                    object_detections_formatted = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}."
                else:
                    object_detections_formatted = f"{unspecific_object_detections_text}."

        # Process object_detections_summary
        if self.use_object_detections_summary and self.object_detections_summary:
            # Replace "video" with "clip"
            summary = self.object_detections_summary.replace("video", "clip")
            if summary.strip():
                if self.lexical_representation != "unformatted":
                    object_detections_summary_formatted = f"Object Detections Summary{double_point}{delimiter}{summary}"
                else:
                    object_detections_summary_formatted = summary

        # Collect all information
        all_information = [
            action_captions_formatted,
            action_captions_summary_formatted,
            object_detections_formatted,
            object_detections_summary_formatted
        ]

        # Filter out None and empty strings
        all_information = [info for info in all_information if info and info.strip()]

        if not all_information:
            return ""

        # Determine level indentation
        if self.lexical_representation == "list":
            level_indentation = "- "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        # Apply indentation
        all_information = [f"{level_indentation}{info}" for info in all_information]

        # Add heading if not unformatted
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)

        # Join all information with line breaks
        lexical_representation = "\n".join(all_information)

        return lexical_representation

    def get_json_representation(self) -> Dict[str, Optional[str]]:
        """
        Generates a JSON representation of the spatial information.

        Returns:
            Dict[str, Optional[str]]: The JSON representation.
        """
        return {
            "action_captions": self.action_captions if self.use_action_captions else None,
            "object_detections": self.object_detections if self.use_object_detections else None,
            "action_captions_summary": self.action_captions_summary if self.use_action_captions_summary else None,
            "object_detections_summary": self.object_detections_summary if self.use_object_detections_summary else None,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_caption_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self) -> str:
        """
        Returns the string representation of the SpatialNodeState.

        Returns:
            str: The lexical representation.
        """
        return self.get_lexical_representation()

    def get_textual_object_list(self) -> List[str]:
        """
        Processes object detections to create a list of formatted strings.

        Returns:
            List[str]: The list of formatted object detection strings.
        """
        unspecific_object_detections_interval_texts = []
        if not self.object_detections:
            return unspecific_object_detections_interval_texts
        
        for objects in self.object_detections:
            if not isinstance(objects, list):
                logger.warning("Each item in object_detections should be a list of objects.")
                continue
            # Capitalize the first letter of each object and strip whitespace
            formatted_objects = [obj.capitalize().strip() for obj in objects if isinstance(obj, str) and obj.strip()]
            if formatted_objects:
                # Join objects with semicolons
                joined_objects = "; ".join(formatted_objects)
                unspecific_object_detections_interval_texts.append(joined_objects)
        
        return unspecific_object_detections_interval_texts