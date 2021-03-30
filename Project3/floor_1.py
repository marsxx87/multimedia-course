import cv2
import numpy as np


def threshold_image(img, threshold):
    ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def filter_dots(img, iters):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, 1)), iterations=iters)


def draw_lines(img, lines):
    for line in lines:
        rad, angle = line[0]
        a, b = np.cos(angle), np.sin(angle)
        x0, y0 = a * rad, b * rad
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def run_2():
    print('project3-2')

    # import and resize image
    floor = cv2.resize(cv2.imread('pic/floor.jpg'), (563, 1000))

    # convert image to grayscale
    floor_grey = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)

    # image binarization by 2 thresholds
    bin1 = threshold_image(floor_grey, 100)
    bin2 = threshold_image(floor_grey, 120)

    # filter some black dots in blocks
    bin1 = filter_dots(bin1, 3)
    bin2 = filter_dots(bin2, 3)

    # blur images
    bin1 = cv2.GaussianBlur(bin1, (9, 9), 0)
    bin2 = cv2.GaussianBlur(bin2, (9, 9), 0)

    # get and draw edge lines from first threshold
    edges = cv2.Canny(bin1, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/2, 250)
    floor = draw_lines(floor, lines)

    # get and draw edge lines from second threshold
    edges = cv2.Canny(bin2, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/2, 250)
    floor = draw_lines(floor, lines)

    # show result image
    cv2.imshow('floor', floor)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

run_2()