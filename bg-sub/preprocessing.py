import cv2 as cv

def preprocessing(img):
    # original phone camera dimensions = 4032 x 3024 (horizontal, no zoom)
    if img is None:
        return None
    x_factor = 0.2
    y_factor = 0.2
    # new shape = 806 x 605
    img = cv.resize(img, None, fx=x_factor, fy=y_factor)
    
    # add denoising if needed
    #img = cv.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
    return img