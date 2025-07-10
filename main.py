import cv2 as cv

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    cv.imshow("video", frame)

    if cv.waitKey(1) == ord('q'):
        break
 