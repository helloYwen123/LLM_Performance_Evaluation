"""Help me to write a class called DeriveRootNodeState, it has a father-class named Operation. and it has following several class member variables :
derive_action_captions: bool, derive_action_captions_summary: bool,num_words_action_captions_summary: int,min_num_words_action_captions_summary: int,derive_object_detections: bool,derive_object_detections_summary: bool,num_words_object_detections_summary: int,min_num_words_object_detections_summary: int,
derive_temporal_grounding: bool,derive_temporal_grounding_summary: bool,num_words_temporal_grounding_summary: int,min_num_words_temporal_grounding_summary: int,normalization_video_length: int = 180
And these member variables are firstly initialized in the __init__ function with passed parameters from __init__().
this class has also two Function members: _execute and get_num_words_for_summary.
\n_execute has given inputs :(graph: Optional[Graph], api: Optional[API], target=Optional[Node]) return nothing
Firstly, it take root node from graph and set it to root_node, compute the length of current video clip, Which is calculated by dividing the length of video clip by the sampled_fps of video clip.
video clip is in the form of 'root_node.state.video_clip'
node this variable has multi-sublevel, using '.' to call ,the form is like root_node.state.spatial_node_state.action_captions. It does also work for root_node.state.video_clip.sampled_fps 
if self.derive_action_caption is true , it will use api's function to get action captions from clips, taking root_node.state.video_clip as input and assign it to variable root_node.state.spatial_node_state.action_captions.
And then output the info and debug in terminal. next if self.derive_action_captions_summary is true, it call api's function get summry_from_noisy_perceptive data() to get variable root_node.state.spatial_node_state.action_captions_summary.
The function called by summary part is named api.get_unspecific_objects_from_video_clip, and it takes action_captions,object_detections,temporal_grounding,interleaved_action_captions_and_object_detections,video_clip,question,words as input.
Apart from action_captions, it's also needed to assign information from api.get_unspecific_objects_from_video_clip to root_node.state.spatial_node_state.object_detections, when if-conditionn is satisfied. Summary part for object dectection is needed and same as action captions.
Additionally, if self.derive_temporal_grounding is true, variable temporal_grounding needs to be assigned via function called api.get_temporal_grounding_from_video_clip_and_text. The results from temporal_grounding has three parts :1.foreground_indicators;2.relevance_indicators3.salience_indicators.You need to respectively save them in node.state
If the self.derive_temporal_grounding_summary is true, it will call api's function get_summary_from_noisy_perceptive_data() to get the summary of temporal grounding.
All of above generated results will be respectively saved into the root_node.state. And set the root_node.state.derived to True.
\nget_num_words_for_summary take len_video_sec;total_num_words_summary;min_num_words_summary as input and return num_words
it does not normalize the number of words for the summary if the normalization video length is -1
this variable: whole_video_word_contingent is equal to total_num_words_summary
otherwise it will normalize the number of words(whole_video_word_contingent),and normalize the number of words for the summary by deviding total_num_words_summary by normalization_video_length and then multiplying the video length in seconds)
and it alse needs to make sure the num of words is multiple of 10 and no less than min_num_words_summary via min python-function. """