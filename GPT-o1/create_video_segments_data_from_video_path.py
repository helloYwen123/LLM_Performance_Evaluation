# GPT-o1
# Ground-Truth
def create_video_segments_data_from_video_path(video_path: str,
                                               sample_rate_scale: int = 1,
                                               new_width: int = 384,
                                               new_height: int = 384,
                                               clip_len: int = 8,
                                               num_segments: int = 1):
    """
    This function calculates a down-sampled representation of a video.
    It does so by splitting the video in #num_segments equally sized segments.
    There are #duration frames sampled from each of these segments.
    Moreover, all these sampled frames can be scaled by sample_rate_scale.

    E.g., if num_segments=10, then a video with 642 frames and 60 fps will be divided into 10 segments each containing
    64 frames. Since this video lasts 642 / 60 = 10.7 seconds which will be floored to 10.0, there is no need to clip
    this value. This is because this value will be used as the number of samples per segment and if the video duration
    is less than 8 seconds, it will always be clipped to 8.0 (to avoid too sparse samples). A linear space with this
    number of indices will then be used to determine the indices for each segment. Afterwards, all those indices are
    concatenated and filtered by the given sample_rate_scale. If it is 1 (per default), all sampled indices will be used
    for sampling. In this example this would result in the following frames:
    [0. 7.1111111   14.2222222  ... 63. 70.1111111  77.2222222  ... 639.]
    With sample_rate_scale = 2.0 this yields (only every second entry from the array above):
    [0. 14.2222222  28.4444444 ... 56.888888   70.1111111   ... 639.]

    Inspired by https://github.com/OpenGVLab/Ask-Anything/blob/long_video_support/video_chat/util.py.

    :param video_path: Path to video.
    :param sample_rate_scale: Number of indices to sample from the array with all indices.
    :param new_width: Scaling of the frame width.
    :param new_height: Scaling of the frame height.
    :param clip_len: Minimum number of samples per segment.
    :param num_segments: Number of segments to sample.
    :return: Numpy video data and JSON metadata.
    """
    # read the video file and get metadata
    vr = VideoReader(video_path, width=new_width, height=new_height, num_threads=1, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    # calculate number of frames per segment (round to floor)
    num_frames_segment = num_frames_total // num_segments

    # calculate duration of the video in seconds
    duration = int(max(num_frames_total // fps, clip_len))

    # use floored duration of the video as number of samples per segment
    samples_per_segment = duration

    # sample as many indices per segment as the video duration
    all_index = []
    for i in range(num_segments):
        index = np.linspace(0, num_frames_segment, num=samples_per_segment)
        index = np.clip(index, 0, num_frames_segment - 1).astype(np.int64)
        index = index + i * num_frames_segment
        all_index.extend(list(index))

    # scale all indices by the given sample rate
    all_index = all_index[::sample_rate_scale]

    # sample the frames
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()

    # create important metadata
    metadata = {
        "video_file": video_path,
        "sample_rate_scale": sample_rate_scale,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": all_index
    }

    # reshape the video data to dimensions (batch, channel, height, width) for convention reasons
    vide_data = torch.from_numpy(buffer)
    vide_data = vide_data.permute(0, 3, 1, 2)

    return vide_data, metadata

# here, you are an very experienced Programmer, who is very good at programming under others' instruction
#  and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# help me to finish a function named create_video_segments_data_from_video_path,    This function calculates a down-sampled representation of a video.
# it takes video_path: str, sample_rate_scale: int = 1, new_width: int = 384,new_height: int = 384,clip_len: int = 8,num_segments: int = 1 as function's inputs
# firstly call VideoReader to get video data as vr, the needed parameters are video_path;new_width: Scaling of the frame width;new_height: Scaling of the frame height;num_threads=1;ctx=cpu(0);
# vr.get_avg_fps() to get fps of video , via len(vr) to return num_frames_total
# calculate number of frames per segments(round to floor) 
# calculate duration of the video in seconds and
# use floored duration of the video as number of samples per segment
# sample as many indices per segment as the video duration
# scale all indices by the given sample rate like using [::sample_rate_scale]
# return to vr start and bases on all_index to get buffer from vr.get_batch, then transfer it to numpy type
# create important metadata, which is dictionary with key and value:
# "video_file": video_path,sample_rate_scale": sample_rate_scale,"fps": fps,"frames_total": num_frames_total,"sample_indices": all_index
# finally reshape the video data to dimensions (batch, channel, height, width) for convention reasons
# and return reshaped vide_data and metadata dictionary

########################################################################################################################################

def create_video_segments_data_from_video_path(
    video_path: str,
    sample_rate_scale: int = 1,
    new_width: int = 384,
    new_height: int = 384,
    clip_len: int = 8,
    num_segments: int = 1
):
    """
    Calculates a down-sampled representation of a video and returns the video data along with metadata.

    Args:
        video_path (str): Path to the video file.
        sample_rate_scale (int, optional): Rate at which frames are sampled. Defaults to 1.
        new_width (int, optional): Desired width of the frames after scaling. Defaults to 384.
        new_height (int, optional): Desired height of the frames after scaling. Defaults to 384.
        clip_len (int, optional): Number of frames per clip. Defaults to 8.
        num_segments (int, optional): Number of segments to divide the video into. Defaults to 1.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the reshaped video data and a metadata dictionary.
    """
    # Initialize VideoReader with specified parameters
    vr = VideoReader(
        video_path,
        width=new_width,
        height=new_height,
        num_threads=1,
        ctx=cpu(0)
    )

    # Get frames per second (FPS) and total number of frames
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    if num_frames_total == 0:
        raise ValueError("The video contains no frames.")

    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")

    # Calculate number of frames per segment (floored)
    frames_per_segment = num_frames_total // num_segments

    if frames_per_segment == 0:
        raise ValueError("Number of segments is greater than the number of frames in the video.")

    # Calculate duration of the video in seconds
    duration = num_frames_total / fps

    # Floored duration as number of samples per segment
    # Assuming duration per segment
    duration_per_segment = duration / num_segments
    samples_per_segment = int(math.floor(duration_per_segment))

    if samples_per_segment == 0:
        raise ValueError("Sample rate scale too high, resulting in zero samples per segment.")

    all_indices = []

    for seg in range(num_segments):
        # Define the start and end frame indices for the current segment
        start_frame = seg * frames_per_segment
        end_frame = start_frame + frames_per_segment

        # Ensure we don't exceed the total number of frames
        end_frame = min(end_frame, num_frames_total)

        # Calculate the number of samples for this segment
        # Here, we sample uniformly across the segment
        if samples_per_segment > (end_frame - start_frame):
            # If samples_per_segment is greater than available frames, adjust it
            current_samples = end_frame - start_frame
        else:
            current_samples = samples_per_segment

        # Generate sample indices within the segment
        # Evenly spaced sampling
        interval = frames_per_segment / current_samples if current_samples > 0 else 1
        for i in range(current_samples):
            idx = int(start_frame + i * interval)
            idx = min(idx, num_frames_total - 1)  # Ensure index is within bounds
            all_indices.append(idx)

    # Apply sample rate scaling
    all_indices = all_indices[::sample_rate_scale]

    if len(all_indices) == 0:
        raise ValueError("No frames selected after applying sample_rate_scale.")

    # Retrieve the frames using the sampled indices
    buffer = vr.get_batch(all_indices).asnumpy()

    # Metadata dictionary
    metadata = {
        "video_file": video_path,
        "sample_rate_scale": sample_rate_scale,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": all_indices
    }

    # Reshape video data to (batch, channel, height, width)
    # Decord returns frames in (batch, height, width, channel)
    video_data = buffer.transpose(0, 3, 1, 2)

    return video_data, metadata

#################################################################################3
# Now i will give you the part of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. Please do not start when i still don't show you my code.

import numpy as np
from decord import VideoReader, cpu
import math

def create_video_segments_data_from_video_path(
    video_path: str,
    sample_rate_scale: int = 1,
    new_width: int = 384,
    new_height: int = 384,
    clip_len: int = 8,
    num_segments: int = 1
):
    """
    This function calculates a down-sampled representation of a video.
    It does so by splitting the video in #num_segments equally sized segments.
    There are #duration frames sampled from each of these segments.
    Moreover, all these sampled frames can be scaled by sample_rate_scale.

    E.g., if num_segments=10, then a video with 642 frames and 60 fps will be divided into 10 segments each containing
    64 frames. Since this video lasts 642 / 60 = 10.7 seconds which will be floored to 10.0, there is no need to clip
    this value. This is because this value will be used as the number of samples per segment and if the video duration
    is less than 8 seconds, it will always be clipped to 8.0 (to avoid too sparse samples). A linear space with this
    number of indices will then be used to determine the indices for each segment. Afterwards, all those indices are
    concatenated and filtered by the given sample_rate_scale. If it is 1 (per default), all sampled indices will be used
    for sampling. In this example this would result in the following frames:
    [0. 7.1111111   14.2222222  ... 63. 70.1111111  77.2222222  ... 639.]
    With sample_rate_scale = 2.0 this yields (only every second entry from the array above):
    [0. 14.2222222  28.4444444 ... 56.888888   70.1111111   ... 639.]

    Inspired by https://github.com/OpenGVLab/Ask-Anything/blob/long_video_support/video_chat/util.py.

    :param video_path: Path to video.
    :param sample_rate_scale: Number of indices to sample from the array with all indices.
    :param new_width: Scaling of the frame width.
    :param new_height: Scaling of the frame height.
    :param clip_len: Minimum number of samples per segment.
    :param num_segments: Number of segments to sample.
    :return: Numpy video data and JSON metadata.
    """
    # Read the video file and get metadata
    vr = VideoReader(video_path, width=new_width, height=new_height, num_threads=1, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    if num_frames_total == 0:
        raise ValueError("The video contains no frames.")

    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")

    # Calculate number of frames per segment (floor division)
    num_frames_segment = num_frames_total // num_segments

    if num_frames_segment == 0:
        raise ValueError("Number of segments is greater than the number of frames in the video.")

    # Calculate duration of the video in seconds and ensure it's at least clip_len
    duration = int(max(num_frames_total / fps, clip_len))

    # Use floored duration of the video as number of samples per segment
    samples_per_segment = duration

    # Initialize list to hold all sampled indices
    all_index = []
    for i in range(num_segments):
        # Generate linearly spaced indices for the current segment
        index = np.linspace(0, num_frames_segment, num=samples_per_segment, endpoint=False)
        index = np.clip(index, 0, num_frames_segment - 1)

        # Offset indices by the start of the current segment
        index = index + i * num_frames_segment

        # Ensure indices are integers and within bounds
        index = np.floor(index).astype(int)
        index = np.clip(index, 0, num_frames_total - 1)

        # Append to the all_index list
        all_index.extend(index.tolist())

    # Convert all_index to a numpy array for efficient slicing
    all_index = np.array(all_index)

    # Apply sample rate scaling
    if sample_rate_scale < 1:
        raise ValueError("sample_rate_scale must be at least 1.")
    all_index = all_index[::sample_rate_scale]

    if len(all_index) == 0:
        raise ValueError("No frames selected after applying sample_rate_scale.")

    # Retrieve the frames using the sampled indices
    buffer = vr.get_batch(all_index).asnumpy()

    # Create metadata dictionary
    metadata = {
        "video_file": video_path,
        "sample_rate_scale": sample_rate_scale,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": all_index.tolist()
    }

    # Reshape video data to (batch, channel, height, width)
    # Decord returns frames in (batch, height, width, channel), so we transpose
    video_data = buffer.transpose(0, 3, 1, 2)

    return video_data, metadata
#####################################################################################################################
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start when i still don't show you my code.

def create_video_segments_data_from_video_path(video_path: str,
                                               sample_rate_scale: int = 1,
                                               new_width: int = 384,
                                               new_height: int = 384,
                                               clip_len: int = 8,
                                               num_segments: int = 1):
    """
    This function calculates a down-sampled representation of a video.
    It does so by splitting the video into `num_segments` equally sized segments.
    There are `duration` frames sampled from each of these segments.
    Moreover, all these sampled frames can be scaled by `sample_rate_scale`.

    E.g., if num_segments=10, then a video with 642 frames and 60 fps will be divided into 10 segments each containing
    64 frames. Since this video lasts 642 / 60 = 10.7 seconds which will be floored to 10.0, there is no need to clip
    this value. This is because this value will be used as the number of samples per segment and if the video duration
    is less than 8 seconds, it will always be clipped to 8.0 (to avoid too sparse samples). A linear space with this
    number of indices will then be used to determine the indices for each segment. Afterwards, all those indices are
    concatenated and filtered by the given sample_rate_scale. If it is 1 (per default), all sampled indices will be used
    for sampling. In this example this would result in the following frames:
    [0. 7.1111111   14.2222222  ... 63. 70.1111111  77.2222222  ... 639.]
    With sample_rate_scale = 2.0 this yields (only every second entry from the array above):
    [0. 14.2222222  28.4444444 ... 56.888888   70.1111111   ... 639.]

    Inspired by https://github.com/OpenGVLab/Ask-Anything/blob/long_video_support/video_chat/util.py.

    :param video_path: Path to video.
    :param sample_rate_scale: Number of indices to sample from the array with all indices.
    :param new_width: Scaling of the frame width.
    :param new_height: Scaling of the frame height.
    :param clip_len: Minimum number of samples per segment.
    :param num_segments: Number of segments to sample.
    :return: Numpy video data and JSON metadata.
    """
    # Read the video file and get metadata
    vr = VideoReader(video_path, width=new_width, height=new_height, num_threads=1, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    # Validate video has frames
    if num_frames_total == 0:
        raise ValueError("The video contains no frames.")

    # Validate number of segments
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")

    # Calculate number of frames per segment (floor division)
    num_frames_segment = num_frames_total // num_segments

    # Validate that each segment has at least one frame
    if num_frames_segment == 0:
        raise ValueError("Number of segments is greater than the number of frames in the video.")

    # Calculate duration of the video in seconds and ensure it's at least clip_len
    duration = int(max(num_frames_total / fps, clip_len))

    # Use floored duration of the video as number of samples per segment
    samples_per_segment = duration

    # Initialize list to hold all sampled indices
    all_index = []
    for i in range(num_segments):
        # Generate linearly spaced indices for the current segment
        index = np.linspace(0, num_frames_segment, num=samples_per_segment, endpoint=False)
        index = np.clip(index, 0, num_frames_segment - 1).astype(np.int64)
        
        # Offset indices by the start of the current segment
        index = index + i * num_frames_segment
        
        # Ensure indices are within the total number of frames
        index = np.clip(index, 0, num_frames_total - 1)
        
        # Append to the all_index list
        all_index.extend(list(index))

    # Convert all_index to a numpy array for efficient slicing
    all_index = np.array(all_index)

    # Apply sample rate scaling
    if sample_rate_scale < 1:
        raise ValueError("sample_rate_scale must be at least 1.")
    all_index = all_index[::sample_rate_scale]

    # Validate that there are indices after scaling
    if len(all_index) == 0:
        raise ValueError("No frames selected after applying sample_rate_scale.")

    # Sample the frames using the sampled indices
    buffer = vr.get_batch(all_index).asnumpy()

    # Create important metadata
    metadata = {
        "video_file": video_path,
        "sample_rate_scale": sample_rate_scale,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": all_index.tolist()
    }

    # Reshape the video data to dimensions (batch, channel, height, width) for convention reasons
    video_data = torch.from_numpy(buffer)
    video_data = video_data.permute(0, 3, 1, 2)  # (batch, channel, height, width)

    return video_data, metadata

