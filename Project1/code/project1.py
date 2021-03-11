import cv2
import numpy as np

img_1 = cv2.imread("pic/holoen.jpg")  #read pic(original)
cv2.imshow("holoen_ori", img_1)   #show pic(original)

height = img_1.shape[0] #read pic height
width = img_1.shape[1]  #read pic width

#-----RGB Channel-----

img_R = np.zeros((height, width, 3), np.uint8)  #create new blank pic
img_R[:, :, 2] = img_1[:, :, 2]                 #get R_ch
cv2.imshow("holoen_R", img_R)                   #show pic_R

img_G = np.zeros((height, width, 3), np.uint8)  #create new blank pic
img_G[:, :, 1] = img_1[:, :, 1]                 #get G_ch
cv2.imshow("holoen_G", img_G)                   #show pic_G

img_B = np.zeros((height, width, 3), np.uint8)  #create new blank pic
img_B[:, :, 0] = img_1[:, :, 0]                 #get B_ch
cv2.imshow("holoen_B", img_B)                   #show pic_B

#-----GRAY SPACE-----
img_grey = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)    #convert img to grey
cv2.imshow("holoen_gray", img_grey)

#-----YCrCb-----
img_ycrcb = cv2.cvtColor(img_1, cv2.COLOR_BGR2YCrCb)
cv2.imshow("holoen_ycrcb", img_ycrcb)

#-----HSV-----
img_hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
cv2.imshow("holoen_hsv", img_hsv)

#-----Shape Drawing-----
img = np.zeros((400, 400, 3), np.uint8)
img.fill(200)

cv2.line(img, (0, 0), (300, 300), (0, 0, 255), 5)
cv2.rectangle(img, (30, 50), (130, 150), (0, 255, 0), 2)
cv2.rectangle(img, (50, 70), (110, 130), (0, 255, 0), -1)
cv2.circle(img, (100, 300), 45, (0, 255, 200), 3)
cv2.circle(img, (200, 220), 60, (255, 0, 0), -1)

cv2.imshow('HW1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = np.zeros((400, 400, 3), np.uint8)

cv2.waitKey(0)