from ultralytics import YOLO

model = YOLO('../runs/detect/train3/weights/best.pt')

results = model('detected/frame188_obj0_Trash_0.61.jpg')

for result in results:
    boxes = result.boxes # Bounding boxes
    masks = result.masks # Segmentation masks
    keypoints = result.keypoints # Keypoints
    probs = result.probs # Class probabilities
    result.show() # Display the results
    #result.save(filename = 'result.jpg')  # Save the results