import cv2 as cv
from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt')
webcam = cv.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, conf = 0.5)

    for result in results:
        annotated_frame = result.plot()
        # cv.imshow("YOLOv9 Detection", frame)
        cv.imshow("YOLOv9 Detection", annotated_frame)

    if cv.waitKey(50) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()