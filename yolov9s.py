from ultralytics import YOLO
import cv2 as cv
import roboflow as rf

# YOLOv9s Example Usage

# Build a YOLOv9c model from scratch
model = YOLO("yolov9s.yaml")

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9s.pt")

# Display model information (optional)
model.info()

# Load a dataset from Roboflow
rf.login()
project = rf.workspace("roboflow-jvuqo").project("plastopol")
dataset = project.version(1).download("yolov9")

# Train the model on the dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLOv9s model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")