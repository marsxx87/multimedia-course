import numpy as np
import cv2 as cv

img = cv.imread('house.jpg')
img = cv.resize(img, (1200, 800))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 1, 0.16)
# result is dilated for marking thr corners, not important
dst = cv.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()