"""Please help me to finish a class named SpatialNodeState it inherit from BaseState. It has several inputs:video_clip: VideoClip,task: Task,lexical_representation: str,use_action_captions: bool,use_object_detections: bool,use_action_captions_summary: bool,use_object_detections_summary: bool
and its fatherclass initialization needs video_clip and task
Its member variables are following:self.lexical_representation;self.action_captions;self.action_captions_summary;self.use_action_captions;self.use_action_captions_summary;self.object_detections;self.object_detections_summary;self.use_object_detections;self.use_object_detections_summary
then report info after Initialization
\nClass's first function is get_lexical_representation, returning string type results
choose the corresponding assignment based on the lexical representation: 
if is "list" then delimiter = " " ;double_point = \":\"
if is "sections" then delimiter = "\n\n";double_point = \"\"
if is "unformatted" then delimiter = \"\";double_point = \"\"
if other types then raise ValueError 
create variable heading f"Spatial Information{{double_point}}"
inititalize action_captions;action_captions_summary;object_detections;object_detections_summary as None
if self.use_action_captions true, then remove the last letter of caption if the last one is .
using '. 'join action_caption
if lexical_representation is not ”unformatted“ , then go to if-branch. If action_caption is not “” then set action_captions to f"Action Captions{{double_point}}{{delimiter}}{{action_captions}}." Otherwise as None
if self.use_action_captions_summary true, then replace "video" words in self.action_captions_summary as "clip"
then  if lexical_representation is not "unformatted"  and if action_captions_summary is not "" , then set action_captions_summary to f"Action Caption Summary{{double_point}}{{delimiter}}{{action_captions_summary}}",Otherwise as None
if self.use_object_detections true, use '. '  to join self.get_textual_object_list() , and assign it to variable unspecific_object_detections_text
then  if lexical_representation is not "unformatted" and if unspecific_object_detections_text != \"\", then set object_detections to f"Object Detections{{double_point}}{{delimiter}}{{unspecific_object_detections_text}}.",Otherwise as None
if self.use_object_detections_summary true,  then then replace "video" words in self.object_detections_summary as \"clip
then  if lexical_representation is not \"unformatted\"  and if object_detections_summary != \"\" , then set object_detections_summary to f"Object Detections Summary{{double_point}}{{delimiter}}{{object_detections_summary}}", otherwise as None
all_information variable is list including elements :action_captions,action_captions_summary,object_detections;object_detections_summary
filter out Nones and empty strings in all_information
if all_information not exists, then return \"\".
choose the corresponding assignment based on the lexical representation: 
if is "list" then level_indentation = \\"
if is "sections" then level_indentation = \"\\n###### \"
if is "unformatted" then level_indentation = \"\"
if other types then raise ValueError 
add corresponding level_indentation with each elements in all_information
if self.lexical_representation != "unformatted" then add heading for all_information
join all_information with "\\n" then assign it to lexical_representation
finally return lexical_representation
\nthe second function is named get_json_representation ,returning dictionary : \"action_captions\": self.action_captions,\"object_detections\": self.object_detections,\"action_captions_summary\": self.action_captions_summary, \"object_detections_summary\": self.object_detections_summary,\"use_action_captions\": self.use_action_captions,\"use_object_detections\": self.use_object_detections,\"use_action_caption_summary\": self.use_action_captions_summary,\"use_object_detections_summary\": self.use_object_detections_summary
\nthe third function __str__ will returns list[str] type variables. And calls function self.get_lexical_representation()
\nthe last function named get_textual_object_list will returns list[str] type variables.
it will respectively pick objects from self.object_detections and then pick object from objects. make first character of each object uppercase and delimit the objects of an interval by semicolons. And remove dots for each object.
returns results named unspecific_object_detections_interval_texts"""