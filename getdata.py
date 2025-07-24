# import os
import roboflow
# import yaml
# import subprocess
# from pathlib import Path
from ultralytics import YOLO

roboflow.login()

rf = roboflow.Roboflow(api_key="biVnTCggzj3GiRiSl5YD")

project = rf.workspace("ed1az").project("plastopol-drizz")
version = project.version(1)
dataset = version.download("yolov9")