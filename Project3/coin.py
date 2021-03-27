import cv2
import numpy as np

def get_color(radius):
    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    if radius < 75:
        return red, 1
    elif radius < 80:
        return orange, 5
    elif radius < 96.5:
        return yellow, 10
    else:
        return green, 50

img = cv2.imread("pic/coin.jpg")
#cv2.imshow("coin_ori", img)
img = cv2.resize(img, (800, 450), interpolation = cv2.INTER_AREA)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("img_resize_gray", img_gray)

ret, coin = cv2.threshold(img_gray, 85, 255, cv2.THRESH_BINARY)     #二值化
#cv2.imshow("coin_binary", coin)

coin = cv2.GaussianBlur(coin, (1, 1), 0)
cv2.imshow("coin1", coin)
coin = cv2.erode(coin, np.ones((2, 2)), iterations = 5)
#cv2.imshow("coin2", coin)
coin = cv2.dilate(coin, np.ones((1, 1)), iterations = 5)
#cv2.imshow("coin3", coin)

ret, coin = cv2.threshold(coin, 180, 255, cv2.THRESH_BINARY)

# connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(coin, connectivity=8, ltype=None)

# draw rectangle around coins and calculate total value
total = 0
for stat in stats:

    x, y, width, height, area = stat
    radius = max(width, height)
    if 50 < radius < 100:

        color, value = get_color(radius)
        total += value
        # draw rectangle around coin
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)

print('total value =', total)

cv2.imshow('coin', img)
cv2.waitKey(0)
cv2.destroyAllWindows()