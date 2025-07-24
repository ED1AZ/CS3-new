import cv2 as cv
import numpy 
from preprocessing import preprocessing

class VideoObject:
    area = 0
    cnt = None
    features = []

    def __init__(self, area, cnt):
        self.area = area
        self.cnt = cnt
        # ADD SELF-DEFINING ATTRIBUTES

    def showBoundary(self, frame):
        x, y, w, h = cv.boundingRect(self.cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, "Moving object", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def extractFeatures(orb, grayframe):
        kp, des = orb.detectAndCompute(grayframe, None)
        img_keypoints = cv.drawKeypoints(grayframe, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow("img", img_keypoints)
        cv.waitKey(0)

    def matchFeatures(prev, next):
        print()

    def defineMotion():
        print()


def bgsubtract(frame, bgsub, learn): 
    blurred = cv.GaussianBlur(frame, (5, 5), 0)

    #0.01 usually
    fgmask = bgsub.apply(blurred, learningRate=learn)
    background_model = bgsub.getBackgroundImage()

    # make kernel bigger for less noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    # open = erodes noise & dilates foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
    # close = fills in holes within foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel, iterations=2)

    return fgmask, background_model

def findObjects(fgmask, frame_gray):
    contours, __ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame_gray, contours, -1, (0,255,0), 3)
    return contours

def defineObjects(contours):
    object_list = []

    def match():
        print()
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter small noise
            # where boundingRect -> putText used to be
            moving_object = VideoObject(area, cnt)
            object_list.append(moving_object)

    return object_list


cam = cv.VideoCapture(0)


bgsub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)

ret, initial = cam.read(0)
initial_mask, background = bgsubtract(initial, bgsub, 1.0)
"""
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cam.read()
old_frame = preprocessing(old_frame)
"""
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame")
    frame = preprocessing(frame)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask, background = bgsubtract(frame, bgsub, 0)
    contours = findObjects(fgmask, frame_gray)

    object_list = defineObjects(contours)
    num = 1
    for i in object_list:
        i.showBoundary(frame)
        print(f"Object #{num}")
        num +=1
    
    cv.imshow("video", fgmask)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv.waitKey(30)  == ord('s'):
        filename = "frames/image" + str(current) + ".jpg"
        cv.imwrite(filename, frame)
        print("Image saved\n")
        current += 1
    elif frame is None:
        break
        
    old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1, 1, 2)
#diff = areasOfInterest(background, frame)
#cv.imshow("diff", diff)
#cv.waitKey(0)
cam.release()
