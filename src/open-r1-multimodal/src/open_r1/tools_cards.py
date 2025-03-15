import sys
import os

import time
import torch
from transformers import pipeline

from PIL import Image, ImageOps

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
sys.path.insert(0, root_dir)
print(root_dir)
from tools.object_detector.tool import Object_Detector_Tool 

tool = Object_Detector_Tool()
metadata = tool.get_metadata()
#print(metadata)

# relative_image_path = "examples/baseball.png"
# data_dir = os.path.join(root_dir,'tools','object_detector')

