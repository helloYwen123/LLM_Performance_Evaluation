import sys
import os


def get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'Tool_Thinker')):
            return os.path.join(current_dir, 'Tool_Thinker')
        current_dir = os.path.dirname(current_dir)
    raise Exception("Could not find project root")

root_dir = get_project_root()
print(os.path.join(root_dir,'tools'))
# add the tools directory to the sys.path
sys.path.insert(0, os.path.join(root_dir, "tools"))

import object_detector as object_detector