import roboflow as roboflow
from dotenv import load_dotenv
from ultralytics import YOLO
import os
import csv

load_dotenv(dotenv_path="keys.env",)
api_key = os.getenv("ED1AZ_API_KEY")

rf = roboflow.Roboflow(api_key=api_key)
# change project & version name
# meant to be used for yolo models 
project = rf.workspace("ed1az").project("weights-pbmkl")
model = project.version(2).model

input_folder = "bg-sub/rois"
output_folder = "output"

# change based off of run
modelName = "YOLOv11s"
LitterNET_used = True
litter_present = True

video_id = str(input("Enter video id: "))

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)

        result = model.predict(image_path=image_path, confidence=70, overlap=30)
        result_json = result.json()
        detections = result_json["predictions"]

        # data format
        # videoID, model, LitterNET_used, litter_present, litter_detected,accuracy, confidence
        if len(detections) > 0:
            print(f"Saved: {filename} (trash detected)")
            for pred in detections:
                conf = pred['confidence']
                class_name = pred['class']

                litter_detected = True
                accuracy = 1 if litter_detected == litter_present else 0

                with open("data_analysis/litternet.csv", 'a') as f:
                    data = [video_id, modelName, LitterNET_used, filename, 0, litter_present, litter_present, conf]
                    writer = csv.writer(f)
                    writer.writerow(data)
                    

            result.save(os.path.join(output_folder, filename))

        else:
            print(f"Skipped: {filename} (no trash detected)")

            
            litter_detected = False
            accuracy = 1 if litter_detected == litter_present else 0
            with open("results.txt", 'a') as f:
                conf = 0
                data = [video_id, modelName, LitterNET_used, filename, 0, litter_present, litter_present, conf]
                writer = csv.writer(f)
                writer.writerow(data)
                 

