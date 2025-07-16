from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict('image3.jpg')

for result in results:
    boxes = result.boxes # Bounding boxes
    masks = result.masks # Segmentation masks
    keypoints = result.keypoints # Keypoints
    probs = result.probs # Class probabilities
    result.show() # Display the results
    result.save(filename = 'result.jpg')  # Save the results