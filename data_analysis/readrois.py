import cv2 as cv
from ultralytics import YOLO
import numpy as np
import csv
import os
import roboflow as roboflow

#OLO_V11S_PATH = '../runs/detect/train3/weights/best.pt'
#model = YOLO(YOLO_V11S_PATH)
api_key = "biVnTCggzj3GiRiSl5YD"#os.getenv("ED1AZ_API_KEY")

rf = roboflow.Roboflow(api_key=api_key)
# change project & version name
project = rf.workspace("ed1az").project("current-dataset-czyp8")
model = project.version(2).model

VIDEO = str(input("Enter video ID: "))
litter_present = True
ROI_PATH = 'rois'
MODEL_TYPE = "YOLOv11s"

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

            results = model.predict(image, confidence=0.5, overlap=0.3).json()


            for object_id, box in enumerate(results['predictions']):
                class_name = box['class']
                conf = box['confidence']

                x, y, width, height = box['x'], box['y'], box['width'], box['height']
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)                      

                litter_detected = True
                data = [VIDEO, MODEL_TYPE, LitterNET, frame_num, object_id, litter_present, litter_detected, conf]
                writer.writerow(data)
                totalDetections += 1

                cv.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                filename = f"detected/frame{frame_num}_obj{object_id}_{class_name}_{conf:.2f}.jpg"
                cv.imwrite(filename, image)

    # Handle case with no detections
    if totalDetections == 0:
        litter_detected = False 
        conf = 0 if litter_present else np.nan
        data = [VIDEO, MODEL_TYPE, LitterNET, np.nan, np.nan, litter_present, litter_detected, conf]
        writer.writerow(data)

cv.destroyAllWindows()