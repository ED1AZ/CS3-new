import cv2 as cv
from preprocessing import preprocessing
import numpy as np
orb = cv.ORB.create()
# VideoEvent : potential litter event or nah?
class VideoEvent:
    potentialLitter = False
    objectPresent = False

    def __init__(self, objectPresent, potentialLitter):
        self.objectPresent = objectPresent
        self.potentialLitter = potentialLitter

    def objectPresence(scene):
        print()
    def litterPresence():
        print()

# VideoObject : has features & motion, wil be tracked as it moves along the screen
# Background Model & FGmask
class VideoObject:
    area = 0
    cnt = None
    features = []
    roi = None

    def __init__(self, area, cnt):
        self.area = area
        self.cnt = cnt
        # ADD SELF-DEFINING ATTRIBUTES

    def showBoundary(self, frame):
        x, y, w, h = cv.boundingRect(self.cnt)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y: y+h, x:x+w]
        cv.putText(frame, "Moving object", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def extractFeatures(self, orb):
        grayroi = cv.cvtColor(self.roi, cv.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(grayroi, None)
        img_keypoints = cv.drawKeypoints(grayroi, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow("img", img_keypoints)
        cv.waitKey(0)

    def matchFeatures(prev, next):
        print()

    def defineMotion():
        print()


def bgsubtract(frame, bgsub): 
    blurred = cv.GaussianBlur(frame, (5, 5), 0)

    # original = 0.01
    fgmask = bgsub.apply(blurred, learningRate=0.01)
    background_model = bgsub.getBackgroundImage()

    # make kernel bigger for less noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    # open = erodes noise & dilates foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel, iterations=2)
    # close = fills in holes within foreground
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel, iterations=2)

    return fgmask, background_model

def areasOfInterest(background, last_frame, round):
    # subtract background & last_frame
    diff = cv.absdiff(background, last_frame)

    # filter out noise & make binary img
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rois = []
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 300 or area < 50:
            # filter out big & small noise
            round += 1
            continue
        else: 
            x, y, w, h = cv.boundingRect(cnt)
            print("Padding:" + str(area))
            padding = int(area//3)
            x_new = max(x - padding, 0)
            y_new = max(y - padding, 0)
            x_end = min(x + w + padding, frame.shape[1])
            y_end = min(y + h + padding, frame.shape[0])

            roi = frame[y_new:y_end, x_new:x_end] 
            rois.append(roi)

    # save rois into folder
    filename = "rois/roi"
    for i, r in enumerate(rois):
        cv.imwrite(filename + str(i+round) + ".png", r)
        print("Saved ROI in " + filename + str(i+round) + ".png")

def findObjects(fgmask, frame_gray):
    contours, __ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(frame_gray, contours, -1, (0,255,0), 3)
    return contours

def defineObjects(contours):
    object_list = []

    def match():
        print()
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter small noise
            moving_object = VideoObject(area, cnt)
            object_list.append(moving_object)
    
    return object_list

def getBackgroundModel(cam, bgsub):
    ret, frame = cam.read()
    __, initial = bgsubtract(frame, bgsub)
    initial = preprocessing(initial)

    return initial


cam = cv.VideoCapture("frames/twotrash.MOV")
#tracker = cv.TrackerCSRT_create()


bgsub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)

initial = getBackgroundModel(cam, bgsub)

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
round = 0
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame")
    frame = preprocessing(frame)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask, background = bgsubtract(frame, bgsub)
    contours = findObjects(fgmask, frame_gray)

    if not contours:
        # contours = empty list --> no objects present
        print("No objects")
        # update background model if no objects
        areasOfInterest(initial, frame, round)
        initial = getBackgroundModel(cam, bgsub)

    else: 
        # for later, extract object features for occlusion/stationary cases
        object_list = defineObjects(contours)
        num = 1
        
        for i in object_list:
            i.showBoundary(frame)
            print(f"Object #{num}")
            num +=1
            print(i.area)
            print(frame.shape)

    
    """
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
 
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        if fgmask[int(b), int(a)] > 0:
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    
    """
    
    #___, threshold = cv.threshold(frame, 250, 255, cv.THRESH_BINARY)
    #threshold = cv.adaptiveThreshold(frame_gray, 0, 255, cv.THRESH_BINARY, 21, 7)
    

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

areasOfInterest(initial, frame, round)

cv.waitKey(0)
cam.release()
