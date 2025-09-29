import cv2 as cv
import os
from skimage.metrics import structural_similarity as ssim

img0 = cv.imread("testing/frame0.png")
img0 = cv.resize(img0, dsize=(1280, 720))
img1 = cv.imread("testing/frame1.png")
img1 = cv.resize(img1, dsize=(1280, 720))


mask0 = cv.imread("testing/mask0.png")
mask0 = cv.resize(mask0, dsize=(1280, 720))
mask1 = cv.imread("testing/mask1.png")
mask1 = cv.resize(mask1, dsize=(1280, 720))

diff = cv.absdiff(img0, img1)
diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
diff = cv.threshold(diff, 60, 255, cv.THRESH_BINARY)[1]

diff2 = cv.absdiff(mask0, mask1)
diff2 = cv.cvtColor(diff2, cv.COLOR_BGR2GRAY)
diff2 = cv.threshold(diff, 60, 255, cv.THRESH_BINARY)[1]

___, binary = cv.threshold(mask0, 200, 1, cv.THRESH_BINARY)
cv.imshow("mask", mask0)
cv.waitKey(0)

cv.imshow("binary", binary)
cv.waitKey(0)

#op = cv.bitwise_xor(diff, diff2)
#op = cv.subtract(diff, diff2)
score, op = ssim(diff, diff2, full=True)
bit = cv.bitwise_xor(diff, diff2)

"""
cv.imshow("frames", diff)
cv.waitKey(0)
cv.imshow("masks", diff2)
cv.waitKey(0)
print(score)
cv.imshow("sim", op)
cv.imshow("xor", bit)
cv.waitKey(0)
"""