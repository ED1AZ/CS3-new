import roboflow as roboflow
from dotenv import load_dotenv
import os
import csv
import cv2 as cv
import numpy as np
import base64


load_dotenv(dotenv_path="keys.env",)
api_key = os.getenv("ED1AZ_API_KEY")

rf = roboflow.Roboflow(api_key=api_key)
# change project & version name
project = rf.workspace("ed1az").project("weights-pbmkl")
model = project.version(1).model

# may have to change video_path
VIDEO = str(input("Enter video ID: "))
VIDEO_PATH = '../bg-sub/frames/' + VIDEO + '.mov'
#VIDEO_PATH = '../bg-sub/frames/001crop.mp4'
output_folder = "detected"
MODEL_TYPE = "RF-DETRs"
LitterNET = False
litter_present = True
frame_num = 0
totalDetections = 0

video = cv.VideoCapture(VIDEO_PATH)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    if frame is None:
        print("No frame")
        break

    results = model.predict(frame, confidence=40, overlap=30).json()
    #results = model(frame)[0]

    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for object_id, box in enumerate(results['predictions']):
            class_name = box['class']
            conf = box['confidence']

            x, y, width, height = box['x'], box['y'], box['width'], box['height']
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)                  

            # videoID, model, LitterNET_used, ___, ____, confidence
            litter_detected = True
            data = [VIDEO, MODEL_TYPE, LitterNET, frame_num, object_id, litter_present, litter_detected, conf]
            writer.writerow(data)
            totalDetections += 1

            # draw rectangle
            cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"Trash: {conf:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            filename = f"detected/frame{frame_num}_obj{object_id}_{conf:.2f}.jpg"
            cv.imwrite(filename, frame)

    frame_num += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# if no objects detected in whole video
if totalDetections == 0:
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        litter_detected = False 

        if litter_present:
            conf = 0 # false negatives
        else:
            conf = np.nan # true negatives
        
        data = [VIDEO, MODEL_TYPE, LitterNET, np.nan, np.nan, litter_present, litter_detected, conf]
        writer.writerow(data)


video.release()
cv.destroyAllWindows()
