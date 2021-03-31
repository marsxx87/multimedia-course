import cv2
import numpy as np

#-----import&resize-----
floor_ori = cv2.imread("pic/floor.jpg")
# cv2.imshow("floor_ori", floor_ori)
floor_resize = cv2.resize(floor_ori, (360, 640), interpolation = cv2.INTER_AREA)
floor_resize0 = cv2.resize(floor_ori, (360, 640), interpolation = cv2.INTER_AREA)
# cv2.imshow("floor_resize0", floor_resize0)

#-----灰階處理-----
floor_gray = cv2.cvtColor(floor_resize, cv2.COLOR_BGR2GRAY)
# cv2.imshow("floor_gray", floor_gray)

#----------去雜訊----------

#-----Blur-----
# floor_blur = cv2.blur(floor_gray, (5, 5))                  #均值濾波
# floor_blur = cv2.GaussianBlur(floor_gray, (3,3), 0)        #高斯濾波
floor_blur = cv2.bilateralFilter(floor_gray, 50, 15, 25)   #雙邊濾波 (臨域直徑, 混和程度, 混和距離)
# floor_blur = cv2.medianBlur(floor_blur, 3)                 #中值濾波
# cv2.imshow("floor_blur1", floor_blur)             

# cv2.imshow("floor_blur2", floor_blur)
# floor_edge = cv2.Canny(floor_blur, 70, 150)
# cv2.imshow("floor_edge", floor_edge)


#-----二值化-----
ret, floor_binary = cv2.threshold(floor_blur, 119, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("floor_binary", floor_binary)

'''
floor_binary = cv2.dilate(floor_binary, np.ones((2, 1)), iterations=1)      #膨脹
cv2.imshow("floor_dilate", floor_binary)
floor_binary = cv2.erode(floor_binary, np.ones((2, 1)), iterations=1)       #侵蝕回去
cv2.imshow("floor_erode", floor_binary)
'''

#----------畫線----------
lines = cv2.HoughLines(floor_binary, 1, np.pi/2, 255)

for line in lines:
    # print(line)
    rad, angle = line[0]
    a, b = np.cos(angle), np.sin(angle)
    x0, y0 = a*rad, b*rad
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 - 1000 *a)
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 + 1000 *a)
    cv2.line(floor_resize, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("floor_resize", floor_resize)


cv2.waitKey(0)
cv2.destroyAllWindows()