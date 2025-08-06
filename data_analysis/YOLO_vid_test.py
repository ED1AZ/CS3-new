import cv2 as cv
from ultralytics import YOLO
import numpy as np
import csv

YOLO_V9S_PATH = '../runs/detect/train3/weights/best.pt'
model = YOLO(YOLO_V9S_PATH)

#str in form of 001, 002, 003, etc
VIDEO = str(input("Enter video ID: "))
litter_present = True
VIDEO_PATH = '../bg-sub/frames/' + VIDEO + '.mov'
video = cv.VideoCapture(VIDEO_PATH)
MODEL_TYPE = "YOLOv9s"

frame_num = 0
totalDetections = 0
LitterNET = False

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)[0]

    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for object_id, box in enumerate(results.boxes):
            class_id = int(box.cls[0])                         
            conf = float(box.conf[0])                        
            class_name = model.names[class_id]                    

            # videoID, model, LitterNET_used, ___, ____, confidence
            litter_detected = True
            data = [VIDEO, MODEL_TYPE, LitterNET, frame_num, object_id, litter_present, litter_detected, conf]
            writer.writerow(data)
            totalDetections += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Save it as an image
            #crop = frame[y1:y2, x1:x2]
            cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"{class_name} {conf:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            filename = f"detected/frame{frame_num}_obj{object_id}_{model.names[class_id]}_{conf:.2f}.jpg"
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


