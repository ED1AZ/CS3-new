import roboflow as roboflow
import supervision as sv
import cv2 as cv
import os 
import pandas as pd
from dotenv import load_dotenv
from iou import iou

FOLDER_PATH = "data_analysis/annotated_frames/test"
W, H = 1920, 1080
global_TP, global_FP, global_FN = 0, 0, 0
detection_records = []  # for CSV


def rfboundingboxcalc(nums_in_line):
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

    return(x_min, y_min, x_max, y_max)


load_dotenv(dotenv_path="keys.env")
api_key = os.getenv("ED1AZ_API_KEY")
rf = roboflow.Roboflow(api_key=api_key)
model = rf.workspace("ed1az").project("current-dataset-czyp8").version(2).model


with os.scandir(FOLDER_PATH + "/images") as images, os.scandir(FOLDER_PATH + "/labels") as labels:
    images = sorted([img for img in images if img.is_file()], key=lambda x: x.name)
    labels = sorted([lbl for lbl in labels if lbl.is_file()], key=lambda x: x.name)

    for frame, label in zip(images, labels):
        true_trash_points = []
        detected_trash_points = []
        detection_confidences = []

        # roboflow ground truth boxes
        with open(label.path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                nums_in_line = [float(item) for item in line.strip().split()]
                x_min, y_min, x_max, y_max = rfboundingboxcalc(nums_in_line=nums_in_line)
                true_trash_points.append([x_min, y_min, x_max, y_max])

        # run model predictions on frame
        img = cv.imread(frame.path)
        results = model.predict(img, confidence=0.5, overlap=0.3).json()
        for box in results.get('predictions', []):
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            conf = box['confidence']
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            detected_trash_points.append([x1, y1, x2, y2])
            detection_confidences.append(conf)

        TP, FP, FN = 0, 0, 0
        matched_preds = set()

        # true negatives
        if not true_trash_points and not detected_trash_points:
            detection_records.append({
                "frame": frame.name,
                "detection_id": None,
                "gt_count": 0,
                "pred_count": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "confidence": None,
                "precision": None,
                "recall": None
            })
            continue

        # false positives
        elif not true_trash_points and detected_trash_points:
            for conf in detection_confidences:
                detection_records.append({
                    "frame": frame.name,
                    "detection_id": None,
                    "gt_count": 0,
                    "pred_count": len(detected_trash_points),
                    "TP": 0,
                    "FP": 1,
                    "FN": 0,
                    "confidence": conf,
                    "precision": None,
                    "recall": None
                })
            global_FP += len(detected_trash_points)
            continue

        # false negative detections
        elif true_trash_points and not detected_trash_points:
            FN = len(true_trash_points)
            global_FN += FN
            detection_records.append({
                "frame": frame.name,
                "detection_id": None,
                "gt_count": len(true_trash_points),
                "pred_count": 0,
                "TP": 0,
                "FP": 0,
                "FN": FN,
                "confidence": None,
                "precision": None,
                "recall": None
            })
            continue

        # match detections to ground truth
        for gt_box in true_trash_points:
            found_match = False
            for i, pred_box in enumerate(detected_trash_points):
                if i in matched_preds:
                    continue
                if iou(gt_box, pred_box, factor=0.5):
                    TP += 1
                    matched_preds.add(i)
                    found_match = True
                    break
            if not found_match:
                FN += 1

        # unmatched detections are false positives 
        for i in range(len(detected_trash_points)):
            if i not in matched_preds:
                FP += 1

        # update global counters
        global_TP += TP
        global_FP += FP
        global_FN += FN

        # per-frame metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None

        # record each detection with confidence
        for i, pred_box in enumerate(detected_trash_points):
            matched = i in matched_preds
            detection_records.append({
                "frame": frame.name,
                "detection_id": i + 1,
                "gt_count": len(true_trash_points),
                "pred_count": len(detected_trash_points),
                "TP": 1 if matched else 0,
                "FP": 0 if matched else 1,
                "FN": 0,
                "confidence": detection_confidences[i],
                "precision": precision,
                "recall": recall
            })

# overall metrics
overall_precision = global_TP / (global_TP + global_FP) if (global_TP + global_FP) > 0 else 0
overall_recall = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

# export
df = pd.DataFrame(detection_records)
csv_path = os.path.join(FOLDER_PATH, "no_litternet1_results.csv")
df.to_csv(csv_path, index=False)

print("\nResults saved to:", csv_path)
print(f"Overall Precision: {overall_precision:.3f}")
print(f"Overall Recall:    {overall_recall:.3f}")
print(f"Overall F1-score:  {overall_f1:.3f}")