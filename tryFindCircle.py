from time import time

import cv2
import numpy as np

test_img = [
    'test_dataset/DSC01313.jpg',
    # 'test_dataset/DSC01332.jpg',
    # 'test_dataset/DSC01422.jpg',
    # 'test_dataset/DSC01426.jpg',
]


def nothing(obj):
    print('in nothing(): obj==%s' % str(obj))


def trackbargui(winname):
    """create trackbars for parameter adjustment"""
    cv2.namedWindow(winname)
    cv2.createTrackbar('R', winname, 0, 255, nothing)
    cv2.createTrackbar('G', winname, 0, 255, nothing)
    cv2.createTrackbar('B', winname, 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, winname, 0, 1, nothing)


def find_circle(img, id):
    winname = "img - "+id
    trackbargui(winname)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pimg = cv2.medianBlur(gray_img, 5)
    cv2.imshow('medianBlur', pimg)
    # pimg = cv2.cvtColor(pimg, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(pimg, cv2.HOUGH_GRADIENT, 1, 120,
                               param1=100, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)

    # cv2.imwrite("1.jpg", img)
    cv2.imshow(winname, img)


def loadimg():
    for i in test_img:
        ori_img = cv2.imread(i)
        ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
        find_circle(ori_img, i)


loadimg()
cv2.waitKey()
cv2.destroyAllWindows()
