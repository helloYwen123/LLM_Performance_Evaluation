# zero-shot instruction
class NodeState(BaseState):
    video_clip: VideoClip
    task: Task
    spatial_node_state: SpatialNodeState
    temporal_node_state: TemporalNodeState
    lexical_representation: str

    def __init__(self, video_clip: VideoClip, task: Task, lexical_representation: str):
        super().__init__(video_clip, task)
        self.ranking = None
        self.ranking_confidence = None
        self.node_state_summary = None
        self.waiting = None
        self.answerable = None
        self.lexical_representation = lexical_representation
        logging.info(
            f"NodeState(video_clip={self.video_clip}, task={self.task}, lexical_representation={self.lexical_representation})"
        )

    def get_lexical_representation(self) -> str:
        if self.lexical_representation == "list":
            prefix = "        -"
            level_indentation = "            - "
            delimiter = ":"
        elif self.lexical_representation == "sections":
            prefix = "\n####"
            level_indentation = "\n##### "
            delimiter = ""
        elif self.lexical_representation == "unformatted":
            prefix = ""
            level_indentation = ""
            delimiter = ""
        else:
            raise ValueError(f"Invalid lexical_representation: {self.lexical_representation}")

        start = self.video_clip.sampled_indices[0] / self.video_clip.original_fps
        end = self.video_clip.sampled_indices[-1] / self.video_clip.original_fps
        all_information = [
            self.spatial_node_state.get_lexical_representation(),
            self.temporal_node_state.get_lexical_representation(),
        ]
        all_information = [
            element for element in all_information if element != ""
        ]  # filter out empty strings
        all_information = [
            level_indentation + element for element in all_information
        ]  # add level_indentation to each element
        if self.lexical_representation != "unformatted":
            all_information = [
                f"{prefix} {self.video_clip.video_name} ({start:.2f}s - {end:.2f}s){delimiter} {element}"
                for element in all_information
            ]  # prepend a heading
        return "\n".join(all_information)

    def get_json_representation(self) -> dict:
        return {
            "video_clip": self.video_clip.get_json_representation(),
            "task": self.task.get_json_representation(),
            "spatial_node_state": self.spatial_node_state.get_json_representation(),
            "temporal_node_state": self.temporal_node_state.get_json_representation(),
            "ranking": self.ranking,
            "ranking_confidence": self.ranking_confidence,
            "node_state_summary": self.node_state_summary,
            "waiting": self.waiting,
            "answerable": self.answerable,
        }

    def merge(self, other: "NodeState") -> "NodeState":
        new_video_clip = self.video_clip.merge(other.video_clip)
        return NodeState(
            new_video_clip,
            self.task,
            self.lexical_representation,
        )

    def __str__(self) -> str:
        return self.get_lexical_representation()
    
# completion 
############################################
class NodeState(BaseState):

    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            spatial_node_state: SpatialNodeState,
            temporal_node_state: TemporalNodeState,
            lexical_representation: str
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.spatial_node_state = spatial_node_state
        self.temporal_node_state = temporal_node_state
        self.lexical_representation = lexical_representation
        self.ranking = None
        self.ranking_confidence = None
        self.node_state_summary = None
        self.waiting = None
        self.answerable = None
        logger.info("NodeState initialized")

    def get_lexical_representation(self) -> str:
        # choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            prefix = "        -"
            level_indentation = "            - "
            delimiter = ":"
        elif self.lexical_representation == "sections":
            prefix = "\n####"
            level_indentation = "\n##### "
            delimiter = ""
        elif self.lexical_representation == "unformatted":
            prefix = ""
            level_indentation = ""
            delimiter = ""
        else:
            raise ValueError("Invalid lexical representation")

        start_time = self.video_clip.sampled_indices[0] / self.video_clip.original_fps
        end_time = self.video_clip.sampled_indices[-1] / self.video_clip.original_fps

        all_information = [
            self.spatial_node_state.get_lexical_representation(),
            self.temporal_node_state.get_lexical_representation()
        ]
        all_information = [info for info in all_information if info]
        all_information = [level_indentation + info for info in all_information]

        if self.lexical_representation!= "unformatted":
            heading = f"{prefix} Information about the clip in the interval [{start_time}, {end_time}]{delimiter}"
            all_information.insert(0, heading)

        return "\n".join(all_information)

    def get_numeric_representation(self) -> torch.Tensor:
        raise NotImplementedError

    def get_json_representation(self) -> dict:
        return {
            "video_clip": self.video_clip.get_json_representation(),
            "task": self.task.get_json_representation(),
            "spatial_node_state": self.spatial_node_state.get_json_representation(),
            "temporal_node_state": self.temporal_node_state.get_json_representation(),
            "lexical_representation": self.lexical_representation,
            "ranking": self.ranking,
            "ranking_confidence": self.ranking_confidence,
            "node_state_summary": self.node_state_summary,
            "waiting": self.waiting,
            "answerable": self.answerable
        }

    def merge(self, other: NodeState) -> NodeState:
        new_video_clip = self.video_clip.get_merged_video_clip(other.video_clip)
        logger.info("Merged node states")

        return NodeState(
            video_clip=new_video_clip,
            task=self.task,
            spatial_node_state=self.spatial_node_state,
            temporal_node_state=self.temporal_node_state,
            lexical_representation=self.lexical_representation
        )

    def __str__(self) -> str:
        return self.get_lexical_representation()
    ###############################################################################
    # completion
    class NodeState(BaseState):

    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            spatial_node_state: SpatialNodeState,
            temporal_node_state: TemporalNodeState,
            lexical_representation: str
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.spatial_node_state = spatial_node_state
        self.temporal_node_state = temporal_node_state
        self.lexical_representation = lexical_representation

        self.ranking = []
        self.ranking_confidence = None
        self.node_state_summary = None
        self.waiting = False
        self.answerable = False

        logger.info(f"Initialized node state")

    def get_lexical_representation(self) -> str:
        # choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            prefix = "        -"
            level_indentation = "            - "
            delimiter = ":"
        elif self.lexical_representation == "sections":
            prefix = "\n####"
            level_indentation = "\n##### "
            delimiter = ""
        elif self.lexical_representation == "unformatted":
            prefix = ""
            level_indentation = ""
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        start = round(self.video_clip.sampled_indices[0] / self.video_clip.original_fps)
        end = round(self.video_clip.sampled_indices[-1] / self.video_clip.original_fps)
        heading = f"{prefix} Information about the clip in the interval [{start}, {end}]{delimiter}"

        # collect all information
        all_information = [
            self.spatial_node_state.get_lexical_representation(),
            self.temporal_node_state.get_lexical_representation()
        ]

        # filter out empty strings
        all_information = [x for x in all_information if x!= ""]

        # add level indentation
        all_information = [level_indentation + x for x in all_information]

        # add heading if not unformatted
        if self.lexical_representation!= "unformatted":
            all_information.insert(0, heading)

        return "\n".join(all_information)

    def get_numeric_representation(self) -> torch.Tensor:
        raise NotImplementedError

    def get_json_representation(self) -> dict:
        return {
            "video_clip": self.video_clip.get_json_representation(),
            "task": self.task.get_json_representation(),
            "spatial_node_state": self.spatial_node_state.get_json_representation(),
            "temporal_node_state": self.temporal_node_state.get_json_representation(),
            "lexical_representation": self.lexical_representation,
            "ranking": self.ranking,
            "ranking_confidence": self.ranking_confidence,
            "node_state_summary": self.node_state_summary,
            "waiting": self.waiting,
            "answerable": self.answerable
        }

    def merge(self, other: NodeState) -> NodeState:
        new_video_clip = self.video_clip.get_merged_video_clip(other.video_clip)
        logger.info("Merged node states")

        return NodeState(
            video_clip=new_video_clip,
            task=self.task,
            spatial_node_state=self.spatial_node_state,
            temporal_node_state=self.temporal_node_state,
            lexical_representation=self.lexical_representation
        )

    def __str__(self):
        return self.get_lexical_representation()