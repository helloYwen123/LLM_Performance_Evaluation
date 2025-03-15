import sys
import os

import time
import torch
import transformers
from transformers import pipeline
import torch.utils.data
from datasets import Dataset, IterableDataset

from PIL import Image, ImageOps

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.insert(0, root_dir)
print(root_dir)
from tools.object_detector.tool import Object_Detector_Tool 

tool = Object_Detector_Tool()
metadata = tool.get_metadata()

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# print(metadata)
# loading model using Qwen2VL (from_pretrained)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
# Processor(from_pretrained)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

relative_image_path = "examples/baseball.png"
image_path = os.path.join(root_dir,'tools','object_detector',relative_image_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
image = Image.open(image_path)


inputs = processor(
    text=[text],
    images=image,
    padding=True,
    return_tensors="pt",
    add_special_tokens=False,
    padding_side="left"
)

inputs = inputs.to("cuda")


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True
)
print(output_text)