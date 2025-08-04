import cv2 as cv

def preprocessing(img):
    # original phone camera dimensions = 4032 x 3024 (horizontal, no zoom)
    x_factor = 0.7
    y_factor = 0.7
    # new shape = 806 x 605
    img = cv.resize(img, None, fx=x_factor, fy=y_factor)
    
    # add denoising if needed
    #img = cv.bilateralFilter(img,9,75,75)
    #img = cv.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21) 

    #img = cv.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
    return img