import cv2 as cv
import numpy as np

cam = cv.VideoCapture("./videos/IMG_1086.mov")

def save_image(frame):
    filename = "images/image.jpg"
    cv.imwrite(filename, frame)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    cv.imshow("Camera Feed", frame)

    key = cv.waitKey(40) & 0xFF # waitKey(1) for a short delay, & 0xFF for cross-platform compatibility
    if key == ord('s'): # Press 's' to save the image
        save_image(frame)
        print("Image saved!")
    elif key == ord('q'): # Press 'q' to quit
        break

cam.release()