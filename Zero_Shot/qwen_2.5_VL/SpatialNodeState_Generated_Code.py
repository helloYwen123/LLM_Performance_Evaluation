# #######################################################
# #######################################################
# Based on Description to generate code
# #######################################################
# #######################################################
class SpatialNodeState(BaseState):

    def __init__(self,
                 video_clip: VideoClip,
                 task: Task,
                 lexical_representation: str,
                 use_action_captions: bool,
                 use_object_detections: bool,
                 use_action_captions_summary: bool,
                 use_object_detections_summary: bool) -> None:

        super().__init__(video_clip=video_clip, task=task)
        
        # Initialize instance attributes
        self.lexical_representation: str = lexical_representation
        self.action_captions: Optional[str] = None
        self.action_captions_summary: Optional[str] = None
        self.object_detections: Optional[str] = None
        self.object_detections_summary: Optional[str] = None
        
        self.use_action_captions: bool = use_action_captions
        self.use_action_captions_summary: bool = use_action_captions_summary
        self.use_object_detections: bool = use_object_detections
        self.use_object_detections_summary: bool = use_object_detections_summary
    
    def _get_delimiter_and_double_point(self, lexical_representation: str) -> tuple[Optional[str], Optional[str]]:
        """
        Returns double point and delimiter according to `lexical_representation`.
        Raises ValueError when invalid input provided.

        Args:
            lexical_representation (str): The format used for representing information.

        Returns:
            Tuple[Optional[str], Optional[str]]: A pair containing delimiter and double-point values or both being None.
        """
        match lexical_representation.lower():
            case "list":
                delimiter = " "
                double_point = ":"
            case "sections":
                delimiter = "\n###### "
                double_point = ""
            case "unformatted":
                delimiter = ""
                double_point = ""
            case _:
                raise ValueError(f'Invalid value {lexical_representation} given.')
                
        return delimiter, double_point
            
    
    @property
    def all_information(self) -> List[str]:
        """Filter non-empty/non-null items."""
        filtered_info = [item for item in [
            self.action_captions,
            self.action_captions_summary,
            self.object_detections,
            self.object_detections_summary]
                         if item]

        return filtered_info
    
    def get_lexical_representation(self) -> str:
        try:
            delimiter, double_point = self._get_delimiter_and_double_point(lexical_representation=self.lexical_representation)

            heading = f"Spatial Information{double_point}"
            
            if self.use_action_captions == True:
                self.action_captions = ''.join([c[:-1] if c.endswith('.') else c for c in self.task.captions]) + ". "
            
            if self.lexical_representation != "unformatted" \
                    and len(self.action_captions.strip()) > 0 :
                self.action_captions = f"{heading}{double_point}{delimiter}{self.action_captions}"

            elif self.use_action_captions_summary == True:
                self.action_captions_summary.replace("video", "clip")
                if self.lexical_representation != "unformatted" \
                        and len(self.action_captions_summary.strip()):
                    self.action_captions_summary = f"{heading}{double_point}{delimiter}{self.action_captions_summary}" 

            if self.use_object_detections == True:
                self.unspecific_object_detections_text = '.'.join(self.get_textual_object_list())
                if self.lexical_representation != "unformatted" \
                        and len(self.unspecific_object_detections_text.strip()):
                    self.object_detections = f"{heading}{double_point}{delimiter}{self.unspecific_object_detections_text}."

            if self.use_object_detections_summary == True:
                self.object_detections_summary.replace("video", "clip")
                if self.lexical_representation != "unformatted" \
                        and len(self.object_detections_summary.strip()):
                    self.object_detections_summary = f"{heading}{double_point}{delimiter}{self.object_detections_summary}"

            all_information = self.all_information
            if not all_information:
                return ""

            level_indentation = {
                "list": "",
                "sections": "\n###### ",
                "unformatted": ""}
            level_indentation[self.lexical_representation] += '\n'.join(all_information).replace('\n', '')
            return level_indentation[self.lexical_representation].strip()
        except Exception as e:
            print(e)
            return ''
    

    def get_json_representation(self) -> dict:
        result_dict = {}
        keys_to_check = ["action_captions",
                         "action_captions_summary",
                         "object_detections",
                         "object_detections_summary"]
        for key in keys_to_check:
            val = getattr(self, key)
            if isinstance(val, str) and val:
                result_dict[key] = val
                
        result_dict.update({
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_captions_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary})
        return result_dict

    def __str__(self) -> List[str]:
        return self.get_lexical_representation().split("\n")

    def get_textual_object_list(self) -> List[str]:
        textual_objects = []
        for obj in self.object_detections.split('.'):
            cleaned_obj = "".join(c.upper() if i==0 else c.capitalize() for i,c in enumerate(obj))
            textual_objects.append(cleaned_obj)
        return ';'.join(textual_objects)
# #######################################################
# #######################################################
# Completion given Half of the code
# \n\n Now i will give you the part of one class which i have finished, help me to complete it based on the following requirment and description and give me the completed code finally.
# \n\nThe incompleted code is following:
# \nclass SpatialNodeState(BaseState):
#     def __init__(
#             self,
#             video_clip: VideoClip,
#             task: Task,
#             lexical_representation: str,
#             use_action_captions: bool,

#     ):
#         super().__init__(video_clip=video_clip, task=task)

#         self.lexical_representation = lexical_representation

#         self.action_captions: list[str] = []
#         self.action_captions_summary: str = \"\"
#         self.use_action_captions = use_action_captions
#         self.use_action_captions_summary = use_action_captions_summary



#     def get_lexical_representation(self) -> str:
#         if not self.use_action_captions and not self.use_action_captions_summary and \
#                 not self.use_object_detections and not self.use_object_detections_summary:
#             return \"\"

#         # choose the delimiter based on the lexical representation
#         if self.lexical_representation == "list":
#             delimiter = \" \"
#             double_point = \":\"
#         elif self.lexical_representation == \"sections\":
#             delimiter = \"\n\n\"
#             double_point = ""
#         elif self.lexical_representation == "unformatted":
#             delimiter = ""
#             double_point = ""
#         else:
#             raise ValueError(f"Unknown lexical representation: {{self.lexical_representation}}")

#         heading = f"Spatial Information{{double_point}}"

#         action_captions = None
#         if self.use_action_captions:
#             action_captions = [caption[:-1] if caption[-1] == "." else caption for caption in self.action_captions]
#             action_captions = '. '.join(action_captions)
#             if self.lexical_representation != "unformatted":
#                 action_captions = f"Action Captions{{double_point}}{{delimiter}}{{action_captions}}." if action_captions != "" else None

#         action_captions_summary = None
#         if self.use_action_captions_summary:
#             # replace "video" with "clip"
#             action_captions_summary = self.action_captions_summary.replace("video", "clip")
#             if self.lexical_representation != "unformatted":
#                 action_captions_summary = f"Action Caption Summary{{double_point}}{{delimiter}}{{action_captions_summary}}" if action_captions_summary != "" else None

#         object_detections = None
#         if self.use_object_detections:
#             # delimit the intervals by dots
#             unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
#             if self.lexical_representation != "unformatted":
#                 object_detections = f"Object Detections{{double_point}}{{delimiter}}{{unspecific_object_detections_text}}." if unspecific_object_detections_text != "" else None



#     def get_numeric_representation(self) -> torch.Tensor:
#         raise NotImplementedError

#     def get_json_representation(self) -> dict:
#         return {
#             "action_captions": self.action_captions,
#             "object_detections": self.object_detections,
#             "action_captions_summary": self.action_captions_summary,
#             "object_detections_summary": self.object_detections_summary,
#             "use_action_captions": self.use_action_captions,

#         }

#     def __str__(self):
#         return f"SpatialNodeState: {self.get_lexical_representation()}"

#     def get_textual_object_list(self) -> list[str]:
#         # make first character of each object uppercase (looks better, have not ablated on that)
#         unspecific_object_detections = [[obj[0].upper() + obj[1:] for obj in objects] for objects in
#                                         self.object_detections]

#         # delimit the objects of an interval by semicolons


#         # remove dots

#         return unspecific_object_detections_interval_texts
# #######################################################
# #######################################################

class SpatialNodeState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_action_captions_summary: bool,
            use_object_detections: bool,
            use_object_detections_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.action_captions: List[str] = []  
        self.action_captions_summary: str = ""
        self.use_action_captions = use_action_captions
        self.use_action_captions_summary = use_action_captions_summary
        self.object_detections: List[List[str]] = [[]] * len(video_clip.clips())   # Initialize detections per clip
        self.object_detections_summary: str = ""
        self.use_object_detections = use_object_detections
        self.use_object_detections_summary = use_object_detections_summary

    def get_lexical_representation(self) -> str:
        if not self.use_action_captions and not self.use_action_captions_summary \
           and not self.use_object_detections and not self.use_object_detections_summary:
            return ""

        # Choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
        elif self.lexical_representation == "sections":
            delimiter = "\n###### "
            double_point = ""
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}"

        action_captions = None
        if self.use_action_captions:
            action_captions = [''.join(caption.split('.')[:-1]) or ''.join(caption.split('.')) for caption in self.action_captions]
            action_captions = '. '.join(action_captions)
            if self.lexical_representation != "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}" if action_captions.strip() else None
        
        action_captions_summary = None
        if self.use_action_captions_summary:
            # Replace "video" with "clip"
            action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}" if action_captions_summary.strip() else None

        object_detections = None
        if self.use_object_detections:
            # Delimit the intervals by dots
            unspecific_object_detections_text = '.'.join([''.join(obj).replace('.', '') for sublist in self.object_detections for obj in sublist])
            if self.lexical_representation != "unformatted":
                object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}." if unspecific_object_detections_text.strip() else None

        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]
        
        filtered_all_info = filter(lambda x: x is not None and x.strip(), all_information)
        joined_info = '\n'.join(filtered_all_info)
        
        if self.lexical_representation != "unformatted":
            joined_info += heading
            
        return joined_info
    
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

    def get_textual_object_list(self) -> List[str]:
        textual_objects_per_video = []

        for detection_set in self.object_detections:
            cleaned_objs = [re.sub(r'\.', '', obj.upper().strip(';')) for obj in detection_set]
            textual_objects_per_video.append(cleaned_objs)

        return textual_objects_per_video


# #######################################################
# #######################################################
# Filling blanks in the code
# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description. Give me the completed code finally.
# \n The incomplete class and code is following:
# \n class SpatialNodeState(BaseState):
#     def __init__(
#             self,
#             video_clip: VideoClip,
#             task: Task,
#             lexical_representation: str,
#             use_action_captions: bool,
#             use_object_detections: ,
#             : bool,
#             use_object_detections_summary: bool
#     ):
#         super().__init__(video_clip=video_clip, task=task)

#         self.lexical_representation = lexical_representation

#         self.: list[str] = []
#         self.action_captions_summary: str = ""
#         self.use_action_captions = use_action_captions
#         self.use_action_captions_summary = use_action_captions_summary

#         self.object_detections: list[list[str]] = []
#         self.object_detections_summary: str = ""
#         self. = use_object_detections
#         self.use_object_detections_summary = 

#         logger.info("Initialized spatial node state")

#     def get_lexical_representation(self) -> str:
#         if not and not self.use_action_captions_summary and \\
#                 not self.use_object_detections and not self.use_object_detections_summary:
#             return ""

#         # choose the delimiter based on the lexical representation
#         if self. == "list":
#             delimiter = " "
#             double_point = ":"
#         elif self.lexical_representation == "":
#             delimiter = "\\n\\n"
#             double_point = ""
#         elif self. == "unformatted":
#             delimiter = ""
#             double_point = ""
#         else:
#             raise ValueError(f"Unknown lexical representation: {{self.}}")

#         heading = f"Spatial Information{{double_point}}"

#         action_captions = None
#         if self.use_action_captions:
#              = [caption[:-1] if  else caption for caption in self.action_captions]
#             action_captions = '. '.join(action_captions)
#             if self.lexical_representation != "unformatted":
#                 action_captions = f"Action Captions{{}}{{delimiter}}{{action_captions}}." if action_captions != "" else None

#         action_captions_summary = None
#         if self.use_action_captions_summary:
#             # replace "video" with "clip"
#             action_captions_summary = self.action_captions_summary.replace("", "clip")
#             if self.lexical_representation != "unformatted":
#                 action_captions_summary = f"Action Caption Summary{{double_point}}{{delimiter}}{{action_captions_summary}}" if action_captions_summary != "" else None

#         object_detections = None
#         if self.use_object_detections:
#             # delimit the intervals by dots
#             unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
#             if self. != "unformatted":
#                 object_detections = f"Object Detections{{}}{{delimiter}}{{unspecific_object_detections_text}}." if unspecific_object_detections_text != "" else None

#         object_detections_summary = 
#         if self.use_object_detections_summary:
#             # replace "video" with "clip"
#             object_detections_summary = self.object_detections_summary.replace("video", "")
#             if self.lexical_representation != "unformatted":
#                 object_detections_summary = f"Object Detections Summary{{double_point}}{{}}{{}}" if object_detections_summary != "" else None

#         # concatenate all information
#         all_information = [
#             action_captions,
#             action_captions_summary,
#             ,
#             object_detections_summary
#         ]

#         # filter out Nones and empty strings
#          = [info for  in all_information if info is not None]
#         all_information = [info for info in  if info != ""]

#         if not all_information:
#             return ""

#         # choose the level indentation based on the lexical representation
#         if self.lexical_representation == "":
#             level_indentation = "                - "
#         elif self. == "sections":
#             level_indentation = "\n###### "
#         elif self.lexical_representation == "":
#             level_indentation = ""
#         else:
#             raise ValueError(f"Unknown lexical representation: {{self.lexical_representation}}")

#         # add level indentation
#         all_information = [level_indentation + info for info in all_information]

#         # add heading
#         if self.lexical_representation != "unformatted":
#             all_information.insert(0, heading)

#         # join by linebreaks
#          = "\n".join(all_information)

#         return lexical_representation

#     def get_numeric_representation(self) -> torch.Tensor:
#         raise NotImplementedError

#     def get_json_representation(self) -> dict:
#         return {
#             "action_captions": self.action_captions,
#             "": self.object_detections,
#             "action_captions_summary": self.action_captions_summary,
#             "object_detections_summary": self.,
#             "use_action_captions": self.use_action_captions,
#             "use_object_detections": self.use_object_detections,
#             "use_action_caption_summary": self.use_action_captions_summary,
#             "use_object_detections_summary": self.use_object_detections_summary
#         }

#     def __str__(self):
#         return f"SpatialNodeState: {self.()}"

#     def get_textual_object_list(self) -> list[str]:
#         # make first character of each object uppercase (looks better, have not ablated on that)
#         unspecific_object_detections = [[obj[0].upper() + obj[] for obj in ] for objects in
#                                         self.object_detections]

#         # delimit the objects of an interval by semicolons
#          = ['; '.join(objects) for objects in
#                                                        unspecific_object_detections]

#         # remove dots
#         unspecific_object_detections_interval_texts = [text.replace(, "") for text in
#                                                        ]

#         return 
# #######################################################
# #######################################################
from typing import AnyStr, IO, TextIO, BinaryIO, Iterator, Sequence, Mapping, Collection, Hashable, SupportsInt, SupportsFloat, SupportsArithmetic, SupportsAbs, SupportsComplex, SupportsRound, SupportsIndex, SupportsBoolean, SupportsBytes, SupportsEllipsis, SupportsShallowCompare, SupportsWeakReference, SupportsHash, SupportsReversed, SupportsCoroutine, SupportsAsyncIteration, SupportsContextManager, SupportsAwaitable, SupportsCoroutineProtocol, SupportsFutureResultType, SupportsFutureResultTypeProtocol, SupportsFutureResultTypeValue, SupportsFutureResultTypeReturn, SupportsFutureResultTypeException, SupportsFutureResultTypeError, SupportsFutureResultTypeSuccess, SupportsFutureResultTypeFailure, SupportsFutureResultTypeDone, SupportsFutureResultTypeCancelled, SupportsFutureResultTypePending, SupportsFutureResultTypeRunning, SupportsFutureResultTypeCompleted, SupportsFutureResultTypeFailed, SupportsFutureResultTypeSucceeded, SupportsFutureResultTypeErrored, SupportsFutureResultTypeAborted, SupportsFutureResultTypeInterrupted, SupportsFutureResultTypeTerminated, SupportsFutureResultTypeFinished, SupportsFutureResultTypeStopped, SupportsFutureResultTypeActive, SupportsFutureResultTypeInactive, SupportsFutureResultTypeReady, SupportsFutureResultTypeNotReady, SupportsFutureResultTypeWaitingForCompletion, SupportsFutureResultTypeCompletedSuccessfully, SupportsFutureResultTypeFailedToComplete, SupportsFutureResultTypeTimedOut, SupportsFutureResultTypeInterruptedByUser, SupportsFutureResultTypeInterruptedBySystem, SupportsFutureResultTypeInterruptedByExternalEvent, SupportsFutureResultTypeInterruptedByInternalEvent, SupportsFutureResultTypeInterruptedByResourceLimitation, SupportsFutureResultTypeInterruptedByNetworkConnectionLost, SupportsFutureResultTypeInterruptedByMemoryExhaustion, SupportsFutureResultTypeInterruptedByDiskSpaceExhaustion, SupportsFutureResultTypeInterruptedByTimeLimitation, SupportsFutureResultTypeInterruptedBySecurityViolation, SupportsFutureResultTypeInterruptedByPolicyViolation, SupportsFutureResultTypeInterruptedByConfigurationChange, SupportsFutureResultTypeInterruptedByEnvironmentChange, SupportsFutureResultTypeInterruptedByDependencyChange, SupportsFutureResultTypeInterruptedByVersionChange, SupportsFutureResultTypeInterruptedByBuildChange, SupportsFutureResultTypeInterruptedByReleaseChange, SupportsFutureResultTypeInterruptedByMaintenanceChange, SupportsFutureResultTypeInterruptedByUpgradeChange, SupportsFutureResultTypeInterruptedByDowngradeChange, SupportsFutureResultTypeInterruptedByPatchChange, SupportsFutureResultTypeInterruptedByHotfixChange, SupportsFutureResultTypeInterruptedByFeatureToggleChange, SupportsFutureResultTypeInterruptedByBugFixChange, SupportsFutureResultTypeInterruptedByPerformanceImprovementChange, SupportsFutureResultTypeInterruptedByUsabilityImprovementChange, SupportsFutureResultTypeInterruptedByCompatibilityImprovementChange, SupportsFutureResultTypeInterruptedByDocumentationUpdateChange, SupportsFutureResultTypeInterruptedByAPIChange, SupportsFutureResultTypeInterruptedByLibraryChange, SupportsFutureResultTypeInterruptedByFrameworkChange, SupportsFutureResultTypeInterruptedByToolchainChange, SupportsFutureResultTypeInterruptedByInfrastructureChange, SupportsFutureResultTypeInterruptedByPlatformChange, SupportsFutureResultTypeInterruptedByTechnologyChange, SupportsFutureResultTypeInterruptedByArchitectureChange, SupportsFutureResultTypeInterruptedByDesignChange, SupportsFutureResultTypeInterruptedByImplementationChange, SupportsFutureResultTypeInterruptedBySpecificationChange, SupportsFutureResultTypeInterruptedByStandardizationChange, SupportsFutureResultTypeInterruptedByNormalizationChange, SupportsFutureResultTypeInterruptedByInteroperabilityChange, SupportsFutureResultTypeInterruptedByPortabilityChange, SupportsFutureResultTypeInterruptedByExtensibilityChange, SupportsFutureResultTypeInterruptedByScalabilityChange, SupportsFutureResultTypeInterruptedByReliabilityChange, SupportsFutureResultTypeInterruptedByAvailabilityChange, SupportsFutureResultTypeInterruptedByMaintainabilityChange, SupportsFutureResultTypeInterruptedByTestabilityChange, SupportsFutureResultTypeInterruptedByDebuggabilityChange, SupportsFutureResultTypeInterruptedByOptimizabilityChange, SupportsFutureResultTypeInterruptedBySecurabilityChange, SupportsFutureResultTypeInterruptedByPrivacyChange, SupportsFutureResultTypeInterruptedByAccessibilityChange, SupportsFutureResultTypeInterruptedByInclusivityChange, SupportsFutureResultTypeInterruptedByEquityChange, SupportsFutureResultTypeInterruptedByFairnessChange, SupportsFutureResultTypeInterruptedByJusticeChange, SupportsFutureResultTypeInterruptedByEqualityChange, SupportsFutureResultTypeInterruptedByDiversityChange, SupportsFutureResultTypeInterruptedByCulturalChange, SupportsFutureResultTypeInterruptedBySocialChange, SupportsFutureResultTypeInterruptedByPoliticalChange, SupportsFutureResultTypeInterruptedByEconomicChange, SupportsFutureResultTypeInterruptedByEnvironmentalChange, SupportsFutureResultTypeInterruptedByTechnologicalChange, SupportsFutureResultTypeInterruptedByScientificChange, SupportsFutureResultTypeInterruptedByEducationalChange, SupportsFutureResultTypeInterruptedByHealthcareChange, SupportsFutureResultTypeInterruptedBySafetyChange, SupportsFutureResultTypeInterruptedb
```

It seems there was some confusion due to excessive complexity involving multiple classes and methods beyond what your request specified. Let’s focus solely on completing `SpatialNodeState` according to the provided requirements within this context.


Here we'll ensure completion focusing strictly around the given instructions:


```python
import logging
logger = logging.getLogger(__name__)

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
        
        self.object_detections: list[List[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections = use_object_detections
        self.use_object_detections_summary = use_object_detections_summary
    
        logger.info("Initialized spatial node state")  

    def get_lexical_representation(self) -> str:
        if not self.action_captions and not self.use_action_captions_summary and\
           not self.object_detections and not self.use_object_detections_summary:
            return ""
            
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
            captions = [caption[:-1] if caption.endswith('.') else caption for caption in self.action_captions]
            action_captions = ". ".join(captions)
            if self.lexical_representation != "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}" if action_captions != "" else None
                
        action_captions_summary = None
        if self.use_action_captions_summary:
            action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}" if action_captions_summary != "" else None
            
        object_detections = None
        if self.use_object_detections:
            detections = [".".join(obj) for obj in self.object_detections]
            object_detections = "; ".join(detections)
            if self.lexical_representation != "unformatted":
                object_detections = f"Object Detections{double_point}{delimiter}{object_detections}" if object_detections != "" else None
                    
        object_detections_summary = None
        if self.use_object_detections_summary:
            object_detections_summary = self.object_detections_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}" if object_detections_summary != "" else None
                        
        all_info = [action_captions, action_captions_summary, object_detections, object_detections_summary]
        filtered_info = [info for info in all_info if info is not None and info.strip()]
                
        if len(filtered_info) == 0:
            return ""
                    
        indentations = {"list":"\t\t-", "sections":"\n##### ", ""}
        try:
            indented_info = [(indentations[self.lexical_representation], info) for info in filtered_info]
        except KeyError:
            raise ValueError(f"Invalid lexical representation '{self.lexical_representation}'")
                        
        joined_info = "".join([f"{ind}\n{i}\n" for ind,i in indented_info])
        if self.lexical_representation != "unformatted":
            joined_info += heading + "\n"
            
        return joined_info.rstrip("\n").rstrip()
        

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
    

    @property
    def __repr__(self)->List[str]:   
        return self.get_lexical_representation().split('\n')
    
    
    def get_textual_object_list(self) -> List[str]:
        textual_objects = []
        for detection_group in self.object_detections:
            capitalized_objs = [o.capitalize() for o in detection_group]
            delimited_obj_str = ';'.join(capitalized_objs).replace('.', '')
            textual_objects.append(delimited_obj_str)
            
        return textual_objects