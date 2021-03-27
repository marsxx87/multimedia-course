import cv2
import numpy as np

#-----import&resize-----
floor_ori = cv2.imread("pic/floor.jpg")
#cv2.imshow("floor_ori", floor_ori)
floor_resize = cv2.resize(floor_ori, (450, 800), interpolation = cv2.INTER_AREA)
#cv2.imshow("floor_resize", floor_resize)

#-----灰階處理-----
floor_gray = cv2.cvtColor(floor_resize, cv2.COLOR_BGR2GRAY)
#cv2.imshow("floor_gray", floor_gray)

#----------去雜訊----------

#-----Blur-----
#floor_blur = cv2.GaussianBlur(floor_gray, (3,3), 0)        #高斯濾波
floor_blur = cv2.medianBlur(floor_gray, 1)                 #中值濾波
#floor_blur = cv2.bilateralFilter(floor_gray, 5, 30, 30)     #雙邊濾波
cv2.imshow("floor_blur", floor_blur)
#floor_edge = cv2.Canny(floor_blur, 70, 150)
#cv2.imshow("floor_edge", floor_edge)


#-----二值化-----
ret, floor_binary = cv2.threshold(floor_blur, 117, 255, cv2.THRESH_BINARY)
cv2.imshow("floor_binary", floor_binary)


#floor_binary = cv2.dilate(floor_binary, np.ones((2, 2)), iterations=1)      #膨脹
#cv2.imshow("floor_dilate", floor_binary)
#floor_binary = cv2.erode(floor_binary, np.ones((1, 1)), iterations=1)       #侵蝕回去
#cv2.imshow("floor_erode", floor_binary)

#----------

lines = cv2.HoughLinesP(floor_binary, 1, np.pi / 180, 100, 100, 10)
#cv2.imshow("", lines)



cv2.waitKey(0)
cv2.destroyAllWindows()