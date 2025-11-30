from ultralytics import YOLO
import os 
import torch
import matplotlib.pyplot as plt
import numpy as np
import json 
import pandas as pd

MODELS = {
    "yolov5": "model_weights/yolov5su.pt",
    "yolov9": "model_weights/yolov9s.pt",
    "yolov11": "model_weights/yolov11s.pt",
    "yolov12": "model_weights/yolov12s.pt"
}

# Model types (each contains k-fold subfolders)
MODEL_TYPES = ["binary"]#, "material", "shape"]

# Training settings
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16


def save_gpu_info(save_dir):
    gpu_info_file = os.path.join(save_dir, "gpu_info.txt")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used_vram = torch.cuda.memory_allocated(0) / (1024**3)
        free_vram = torch.cuda.memory_reserved(0) / (1024**3)
    else:
        gpu_name = "CPU"
        total_vram = used_vram = free_vram = 0

    with open(gpu_info_file, "w") as f:
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Total VRAM: {total_vram:.2f} GB\n")
        f.write(f"Used VRAM: {used_vram:.2f} GB\n")
        f.write(f"Free VRAM: {free_vram:.2f} GB\n")
        f.write(f"Batch Size: {BATCH}\n")
        f.write(f"Epochs: {EPOCHS}\n")

def save_confusion_matrix(cm, labels, save_path):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


for model_type in MODEL_TYPES:

    if not os.path.isdir(model_type):
        print(f"Skipping {model_type}, folder not found.")
        continue

    for model_name, weight_file in MODELS.items():

        print(f"\nTraining: {model_type} | {model_name} ===")

        for fold in os.scandir(model_type):
            if not fold.is_dir():
                continue
            
            model = YOLO(weight_file)

            data_yaml = os.path.join(fold.path, "data.yaml")

            save_dir = f"results/{model_type}/{model_name}/fold_{fold.name}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\nFOLD: {fold.name}")
            print(f"  Data: {data_yaml}")

            # ------------------------------
            # Train
            # ------------------------------
            model.train(
                data=data_yaml,
                epochs=EPOCHS,
                batch=BATCH,
                project=save_dir,
                name=".",
                exist_ok=True
            )

            # ------------------------------
            # Validate to gather metrics
            # ------------------------------
            metrics = model.val(data=data_yaml)
            metrics_dict = metrics.results_dict

            # Save raw metrics JSON
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics_dict, f, indent=4)

            # Confusion matrix
            cm = metrics.confusion_matrix.matrix
            class_names = metrics.names

            save_confusion_matrix(
                cm, 
                list(class_names.values()),
                os.path.join(save_dir, "confusion_matrix.png")
            )

            # ------------------------------
            # Class-wise metrics (YOLOv8)
            # ------------------------------
            tp = metrics.box.tp
            fp = metrics.box.fp
            fn = metrics.box.fn

            # Precision per class (manual)
            precision = tp / (tp + fp + 1e-16)

            # Recall per class (native)
            recall = metrics.box.recall_per_class

            # F1 score per class (manual)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

            # AP metrics
            ap50 = metrics.box.map50_per_class      # mAP@0.50
            ap5095 = metrics.box.map_per_class      # mAP@0.50–0.95

            # Save class-wise table
            df = pd.DataFrame({
                "class_id": list(class_names.keys()),
                "class_name": list(class_names.values()),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "ap50": ap50.tolist(),
                "ap50-95": ap5095.tolist()
            })

            df.to_csv(os.path.join(save_dir, "metrics_table.csv"), index=False)

            # ------------------------------
            # Inference speeds
            # ------------------------------
            speed = metrics.speed

            with open(os.path.join(save_dir, "inference_speed.json"), "w") as f:
                json.dump({
                    "preprocess_ms": speed["preprocess"],
                    "inference_ms": speed["inference"],
                    "postprocess_ms": speed["postprocess"],
                    "total_ms": sum(speed.values())
                }, f, indent=4)

            save_gpu_info(save_dir)