import cv2
import numpy as np

img = cv2.imread("pic/man.jpg", 0)
ret, base = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, inversed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

inversed = cv2.erode(inversed, np.ones((3, 3)), iterations = 3)

hw2_man = cv2.subtract(base, inversed)

#cv2.imshow('base', base)
#cv2.imshow('inversed', inversed)
cv2.imshow('HW2_man', hw2_man)
cv2.waitKey(0)
cv2.destroyAllWindows()