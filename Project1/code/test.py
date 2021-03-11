import cv2
import numpy as np

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