import cv2 as cv
from ultralytics import YOLO

webcam = cv.VideoCapture(0)

model = YOLO('runs/detect/train/weights/best.pt')

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(frame)

    cv.imshow("result", results)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()