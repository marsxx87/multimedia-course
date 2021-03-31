import cv2
import numpy as np

def get_coin_color(diam):
    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    if diam < 58:
        return red, 1
    elif diam < 65:
        return orange, 5
    elif diam < 75:
        return yellow, 10
    else:
        return green, 50

def get_banknote_color(rgb):
    blue = (255, 0, 0)
    purple = (240, 32, 160)
    white = (255, 255, 255)

    if rgb[0] > rgb[1] and rgb[1] > rgb[2]:
        return white, 1000
    elif rgb[2] > 190:
        return blue, 100
    else:
        return purple, 500

img = cv2.imread("pic/coin2.jpg")
img = cv2.resize(img, (1000, 563), interpolation = cv2.INTER_AREA)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, coin = cv2.threshold(img_gray, 85, 255, cv2.THRESH_BINARY)

coin = cv2.GaussianBlur(coin, (1, 1), 0)
coin = cv2.erode(coin, np.ones((2, 2)), iterations = 2)

# connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(coin, connectivity = 8, ltype = None)

# draw rectangle around coins and calculate total value
total = 0
for stat in stats:

    x, y, width, height, area = stat
    diam = max(width, height)
    if 50 < diam < 100:
        color, value = get_coin_color(diam)
        total += value
        # draw rectangle around coin
        cv2.rectangle(img, (x, y), (x + diam, y + diam), color, 2)

    elif 100 <= diam < 500:
        rgb = [round(np.average(img[y: y + height, x: x + width, k])) for k in range(3)]
        color, value = get_banknote_color(rgb)
        total += value
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)


print('total value =', total)

cv2.imshow('coin2', img)
cv2.waitKey(0)
cv2.destroyAllWindows()