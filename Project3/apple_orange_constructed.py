import cv2
import numpy as np

apple = cv2.imread("pic/apple.jpg")
orange = cv2.imread("pic/orange.jpg")

# Guassian Pyramids
apple_copy = apple.copy()
apple_gaussian = [apple_copy]

for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    apple_gaussian.append(apple_copy)

orange_copy = orange.copy()
orange_gaussian = [orange_copy]

for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    orange_gaussian.append(orange_copy)

# Laplacian Pyramids
apple_copy = apple_gaussian[5]
apple_laplacian = [apple_copy]

for i in range (5,0,-1):
    gaussian_extended = cv2.pyrUp(apple_gaussian[i])
    
    laplacian = cv2.subtract(apple_gaussian[i-1], gaussian_extended)
    apple_laplacian.append(laplacian)

orange_copy = orange_gaussian[5]
orange_laplacian = [orange_copy]

for i in range (5,0,-1):
    gaussian_extended = cv2.pyrUp(orange_gaussian[i])

    laplacian = cv2.subtract(orange_gaussian[i-1], gaussian_extended)
    orange_laplacian.append(laplacian)

# join
apple_orange_pyramid = []
n = 0
for apple_lp, orange_lp in zip(apple_laplacian, orange_laplacian):
    n += 1
    cols, rows, ch = apple_lp.shape
    laplacian = np.hstack((apple_lp[:, 0:int(cols/2)], orange_lp[:, int(cols/2):]))
    apple_orange_pyramid.append(laplacian)
    
# reconstrut image
img_reconstruct = apple_orange_pyramid[0]
for i in range(1,6):
    img_reconstruct = cv2.pyrUp(img_reconstruct)
    img_reconstruct = cv2.add(apple_orange_pyramid[i], img_reconstruct)


cv2.imshow('apple_orange_reconstruct', img_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()