import roboflow as roboflow
import supervision as sv
import cv2 as cv
import os 
from dotenv import load_dotenv
import pandas as pd

csv_file = "data_analysis/litternet.csv" # csv path

def update_csv(row_number, update_dict):

    # csv path
    df = pd.read_csv(csv_file)
    if row_number < 0 or row_number >= len(df):
        raise IndexError(f"{row_number} out of range")

    # update w for loop through dict
    for col, val in update_dict.items():
        if col in df.columns:
            df.at[row_number, col] = val
        else:
            print(f"{col} not found")

    df.to_csv(csv_file, index=False)

    print(f"Row {row_number} updated successfully in {csv_file}.")

update_csv(1, {"VIDEO": 1, "MODEL_TYPE": "YOLO", 
               "LitterNET": False, "frame_num": 
               "roi0.png", "object_id": 69, 
               "litter_present": False, 
               "litter_detected": False, "conf": 3})

load_dotenv(dotenv_path="keys.env")
api_key = os.getenv("ED1AZ_API_KEY")
rf = roboflow.Roboflow(api_key=api_key)
model = rf.workspace("ed1az").project("current-dataset-czyp8").version(2).model

img = cv.imread("data_analysis/annotated_frames/test/images/013_MOV-0008_jpg.rf.a241f21bedcfed362b3f6626471318dc.jpg")
#results = model.predict(img, confidence=40, overlap=30)


#model = inference.get_model(model_id="current-dataset-czyp8/2", api_key=api_key)
"""
# finds bounding box of model prediction
results = model.predict(img, confidence=0.5, overlap=0.3).json()
for object_id, box in enumerate(results['predictions']):
    class_name = box['class']
    conf = box['confidence']

    x, y, width, height = box['x'], box['y'], box['width'], box['height']
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)    
    print(x1, y1, x2, y2)

detections = sv.Detections.from_inference(results)
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
cv.imwrite("annotated.jpg", annotated_image)
"""
# collect bounding box from correlating frame

with open("data_analysis/annotated_frames/test/labels/013_MOV-0008_jpg.rf.a241f21bedcfed362b3f6626471318dc.txt", 'r') as file:
    for line in file:
        nums_in_line = line.strip().split()
        nums_in_line = [float(item) for item in nums_in_line]

W, H = 1920, 1080
__, x_center_norm, y_center_norm, width_norm, height_norm = nums_in_line
x_center_pixel = x_center_norm * W
y_center_pixel = y_center_norm * H
width_pixel = width_norm * W
height_pixel = height_norm * H
 
#top left corner
x_min = x_center_pixel - width_pixel / 2
y_min = y_center_pixel - height_pixel / 2
x_max = x_center_pixel + width_pixel / 2
y_max = y_center_pixel + height_pixel / 2

print(x_min, y_min, x_max, y_max)
#x1, y1, x2, y2 = points to compare to