import roboflow as roboflow
from dotenv import load_dotenv
import os
import cv2

video_path = "bg-sub/frames/001.mp4"

# Count frames Roboflow extracted
num_frames = 20

# Check duration of the original video
cap = cv2.VideoCapture(video_path)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"Frames extracted: {num_frames}")
print(f"Duration: {duration:.2f} seconds")
print(f"Effective sampling FPS: {num_frames / duration:.2f}")