# import os
import roboflow
from dotenv import load_dotenv
import os
# import yaml
# import subprocess
# from pathlib import Path
# from ultralytics import YOLO

load_dotenv(dotenv_path="keys.env",)
api_key = os.getenv("CS3_API_KEY")
rf = roboflow.Roboflow(api_key=api_key)

project = rf.workspace("cs3-ug7j5").project("litter-detection-pbz2j")
version = project.version(6)
dataset = version.download("yolov9")