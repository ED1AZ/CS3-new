from ultralytics import YOLO
import time
import statistics
import glob
import os

# Path to the folder with images
image_folder = "parklitter_jpg/"  # make sure it ends with /
image_paths = glob.glob(os.path.join(image_folder, "*"))

# Load your YOLO model
model = YOLO('./weights.pt')

# Store inference times (in milliseconds)
times = []

for img_path in image_paths:
    start_time = time.time()  # start timer
    
    results = model(img_path)  # run detection
    
    end_time = time.time()  # end timer
    elapsed_ms = (end_time - start_time) * 1000  # convert to milliseconds
    
    times.append(elapsed_ms)
    print(f"{os.path.basename(img_path)}: {elapsed_ms:.2f} ms")

# Compute average time
if times:
    avg_time = statistics.mean(times)
    print(f"\naverage inference time across {len(times)} images: {avg_time:.2f} ms")
else:
    print("\nno images found")