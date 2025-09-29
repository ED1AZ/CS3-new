import cv2 as cv
import os

path = "bg-sub/frames/long.mp4" #change later
cam = cv.VideoCapture(path)
bgsub = cv.createBackgroundSubtractorMOG2(detectShadows=False)

round = 2
num = 0

# Make dynamic background dataset with bgsub
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame")
        break

    mask = bgsub.apply(frame)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel=kernel)

    
    cv.imshow("Mask", mask)

    """
    if num == 1:
        cv.imwrite("testing/mask0.png", mask)
        cv.imwrite("testing/frame0.png", frame)

    elif num == 2:
        cv.imwrite("testing/mask1.png", mask)
        cv.imwrite("testing/frame1.png", frame)

    num+=1
    """

    if cv.waitKey(1) == ord('q'):
        print("Exiting..")
        break
    elif cv.waitKey(1) == ord('s'):
        cv.imwrite("testing/mask" + str(round) + ".png", mask)
        cv.imwrite("testing/frame"+ str(round) +".png", frame)
        round += 1


cam.release()
cv.destroyAllWindows()