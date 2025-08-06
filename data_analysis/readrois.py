import cv2 as cv
from ultralytics import YOLO
import numpy as np
import csv
import os

YOLO_V11S_PATH = '../runs/detect/train3/weights/best.pt'
model = YOLO(YOLO_V11S_PATH)

VIDEO = str(input("Enter video ID: "))
litter_present = True
ROI_PATH = '../bg-sub/rois'
MODEL_TYPE = "YOLOv9s"

frame_num = 0
totalDetections = 0
LitterNET = False

# Output folder
#os.makedirs("detected", exist_ok=True)

# Open CSV for writing results
with open('litternet.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for roi_file in os.listdir(ROI_PATH):
        if roi_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Running on: {roi_file}")
            roi_path = os.path.join(ROI_PATH, roi_file)
            image = cv.imread(roi_path)

            if image is None:
                print(f"Failed to load image: {roi_path}")
                continue

            results = model(image)

            for object_id, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])                         
                conf = float(box.conf[0])                        
                class_name = model.names[class_id]                    

                litter_detected = True
                data = [VIDEO, MODEL_TYPE, LitterNET, frame_num, object_id, litter_present, litter_detected, conf]
                writer.writerow(data)
                totalDetections += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                label = f"{class_name} {conf:.2f}"
                cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                filename = f"detected/frame{frame_num}_obj{object_id}_{class_name}_{conf:.2f}.jpg"
                cv.imwrite(filename, image)
            
            if not results[0].boxes:
                data = [VIDEO, MODEL_TYPE, LitterNET, frame_num, object_id, False, False, 0]
                writer.writerow(data)
            frame_num += 1

    # Handle case with no detections
    if totalDetections == 0:
        litter_detected = False 
        conf = 0 if litter_present else np.nan
        data = [VIDEO, MODEL_TYPE, LitterNET, np.nan, np.nan, litter_present, litter_detected, conf]
        writer.writerow(data)

cv.destroyAllWindows()