import cv2 as cv
from preprocessing import preprocessing

class VideoObject:
    def showBoundary():
        print()
    def extractFeatures(sift, grayframe):
        kp1 = sift.detect(grayframe, None)
        img_keypoints = cv.drawKeypoints(grayframe, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow("img", img_keypoints)
        cv.waitKey(0)
    def matchFeatures(prev, next):
        print()

"""
sift = cv.SIFT()
kp1, des1 = sift.detectAndCompute()
#img_keypoints = cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bf = cv.BFMatcher()
"""

def bgsubtract(frame, bgsub): 
    frame = preprocessing(frame)
    blurred = cv.GaussianBlur(frame, (5, 5), 0)
    fgmask = bgsub.apply(blurred, learningRate=0.01)

    # make kernel bigger for less noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    # open = erodes noise & dilates foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
    # close = fills in holes within foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel, iterations=2)

    return fgmask



cam = cv.VideoCapture("frames/park-walk.mp4")
bgsub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame")
    frame = bgsubtract(frame, bgsub)
    
    

    cv.imshow("video", frame)
    
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
    elif cv.waitKey(1)  == ord('s'):
        filename = "frames/image" + str(current) + ".jpg"
        cv.imwrite(filename, frame)
        print("Image saved\n")
        current += 1
