#電資二 108820006 楊品賢
#電資二 108820018 蔡翔宇

import cv2
import numpy as np

img_circle = cv2.imread("pic/circle.jpg", 0)    #read pic in gray
cv2.imshow("circle_ori", img_circle)           #show original pic

ret, out1 = cv2.threshold(img_circle, 127, 255, cv2.THRESH_OTSU)  #二值化
#cv2.imshow("BINARY", out1)                                       #show pic

out2 = cv2.dilate(out1, np.ones((5, 5)), iterations=4)      #膨脹
out3 = cv2.erode(out2, np.ones((5, 5)), iterations=4)       #侵蝕回去
cv2.imshow("out3", out3)


img_man = cv2.imread("pic/man.jpg", 0)
cv2.imshow("man_ori", img_man)                                      #show man
ret, base = cv2.threshold(img_man, 127, 255, cv2.THRESH_BINARY)
ret, inversed = cv2.threshold(img_man, 127, 255, cv2.THRESH_BINARY)

inversed = cv2.erode(inversed, np.ones((3, 3)), iterations = 3)

hw2_man = cv2.subtract(base, inversed)

#cv2.imshow('base', base)
#cv2.imshow('inversed', inversed)
cv2.imshow('HW2_man', hw2_man)

cv2.waitKey(0)