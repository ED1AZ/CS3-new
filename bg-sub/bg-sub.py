import cv2 as cv
from preprocessing import preprocessing
import numpy as np
import os
from iou import iou

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

def findObjects(fgmask, mode=cv.RETR_EXTERNAL):
    contours, hierarchy = cv.findContours(fgmask, mode, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(frame_gray, contours, -1, (0,255,0), 3)
    return contours, hierarchy


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


def saveROI(rois, roundnum):
    filename = "rois/roi"

    for i, r in enumerate(rois):
        cv.imwrite(filename + str(i+roundnum) + ".png", r)
        print("Saved ROI in " + filename + str(i+roundnum) + ".png")


def findBinaryDiff(background, last_frame): 
    # subtract background & last_frame
    diff = cv.absdiff(background, last_frame)
    
    # make binary img
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]

    # erode noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

    return thresh 

def eliminateDuplicates(rois):
    rois = sorted(rois, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]), reverse=True)
    precision = 5
    if precision > 0:
        rois = [[round(v / precision) * precision for v in box] for box in rois]

    filtered = []

    for i, boxA in enumerate(rois):
        duplicate = False
        for boxB in filtered:
            if iou((boxA), (boxB)):
                duplicate = True
                break
        if not duplicate:
            filtered.append(boxA)

    return filtered

def isolateGround(background, last_frame):
    height, width, __ = background.shape
    ground = int(height/2)

    # keep bottom part of image
    background = background[ground:height, 0:width]
    last_frame = last_frame[ground:height, 0:width]

    return background, last_frame

def areasOfInterest(background, last_frame, roundnum):
    background, last_frame = isolateGround(background, last_frame)
    
    diff = findBinaryDiff(background, last_frame)
    static_changes, ___ = findObjects(diff)
    rois = []
    points = []

    for ch in static_changes:
        area = cv.contourArea(ch)
        if area > 6000 or area < 50:
            # filter out big & small noise
            print(f"{area}: Change too big or too small")
            continue
        else: 
            x, y, w, h = cv.boundingRect(ch) # coordinates in original half-frame
            padding = 50 #int(area//20)

            # coordinates to crop
            x_new = int(max(x - padding, 0))
            y_new = int(max(y - padding, 0))
            x_end = int(min(x + w + padding, last_frame.shape[1]))
            y_end = int(min(y + h + padding, last_frame.shape[0]))

            """
            roi = last_frame[y_new:y_end, x_new:x_end] 
            #roi = cv.rectangle(roi, (x-x_new, y-y_new), (x-x_new + w, y-y_new + h), (0, 0, 255), 2)
            rois.append(roi)
            """
            points.append([x_new, y_new, x_end, y_end])

    roi_points = eliminateDuplicates(points)

    for group in roi_points:
        roi = last_frame[group[1]:group[3], group[0]:group[2]]
        #roi = cv.rectangle(roi, (x-x_new, y-y_new), (x-x_new + w, y-y_new + h), (0, 0, 255), 2)
        rois.append(roi)

    if not rois:
        print("No static changes in scene")
    else: 
        saveROI(rois, roundnum)
        roundnum += 1

    return diff

def defineObjects(contours):
    object_list = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter small noise (500 originally)
            moving_object = VideoObject(area, cnt)
            object_list.append(moving_object)
    
    return object_list

def getBackgroundModel(cam, bgsub):
    ret, frame = cam.read()
    __, initial = bgsubtract(frame, bgsub)
    #initial = preprocessing(initial)

    return initial

# playing.mov error bc object never leaves frame
# rolling.mov error bc bottle rolls out of frame, so its img is replaced
# table.mov error bc bottle never detected?
# two bystander.mov error bc WAY too many ROIS generated

cam = cv.VideoCapture("frames/occlusion1.mov")

bgsub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)
#bgsub = cv.createBackgroundSubtractorKNN(history=20, dist2Threshold=50, detectShadows=False)

initial = getBackgroundModel(cam, bgsub)

roundnum = 0
current = 0
avg = initial.copy()

if os.path.isdir("rois"):
    os.rmdir("rois")
os.mkdir("rois")

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame")
        break
    if frame is None:
        break
    else:
        # video var is for viewing, frame is for processing
        video = preprocessing(frame)

    #frame = cv.convertScaleAbs(frame, alpha=1, beta=-20)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask, ___ = bgsubtract(frame_gray, bgsub)

    alpha = 0.2
    beta = 1 - alpha
    avg = cv.addWeighted(avg, alpha, frame, beta, 0.0)
    
    moving_contours, ___  = findObjects(fgmask)

    if not moving_contours:
        # moving_contours = empty list --> no objects present
        print("No objects")

        # update background model if no objects
        diff = areasOfInterest(initial, avg, roundnum)
        
        #initial = getBackgroundModel(cam, bgsub)

    else: 
        # for later, extract object features for occlusion/stationary cases
        object_list = defineObjects(moving_contours)
        num = 1
        
        for i in object_list:
            i.showBoundary(frame)
            print(f"Object #{num}")
            num +=1
            print(i.area)

        
    #fgmask or video for viewing
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

cam.release() 