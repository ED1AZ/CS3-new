import cv2 as cv
import os
"""
manitou-a = 2
trout-a = 3
trout-c = 4
"""
bgsub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
round = 0
num = 0

# Cycle through background motion videos in folder
for video in os.listdir("testing/videos"):
    path = "testing/videos/video" + str(round) + ".mp4"
    cam = cv.VideoCapture(path)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't read frame")
            break

        mask = bgsub.apply(frame)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel=kernel)
        cv.imshow("Mask", mask)


        if cv.waitKey(1) == ord('q'):
            print("Exiting..")
            break
        
        
        cv.imwrite("testing/dataset/train/masks/" + str(round) + "mask" + str(num) + ".jpg", mask)
        cv.imwrite("testing/dataset/train/frames/" + str(round) + "frame"+ str(num) +".jpg", frame)
        num += 1
    round += 1

    cam.release()

cv.destroyAllWindows()

#after getting all masks, shuffle them between test, train, & val folders