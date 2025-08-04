import roboflow as roboflow
from dotenv import load_dotenv
import supervision as sv
import os

load_dotenv(dotenv_path="keys.env",)
api_key = os.getenv("CS3_API_KEY")

rf = roboflow.Roboflow(api_key=api_key)
project = rf.workspace("cs3-ug7j5").project("litter-detection-pbz2j")
model = project.version(6).model

input_folder = "bg-sub/rois"
output_folder = "output"

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)

        result = model.predict(image_path, confidence=70, overlap=30)
        result_json = result.json()
        detections = result_json["predictions"]


        if len(detections) > 0:
            print(f"Saved: {filename} (trash detected)")
            for pred in detections:
                conf = round(pred['confidence'] * 100, 2)
                class_name = pred['class']
                print(f" - {class_name}: {conf}%")

                with open("results.txt", 'w') as f:
                    f.write(filename + " ")
                    f.write(str(conf))
                    f.write("\n")

            result.save(os.path.join(output_folder, filename))

        else:
            print(f"Skipped: {filename} (no trash detected)")


