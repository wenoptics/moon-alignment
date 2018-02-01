from time import time

import cv2
import numpy as np

from utils.util import myTuneWindow


class Circle():
    def __init__(self, x, y, r):
        self.r = r
        self.y = y
        self.x = x

# Normalize image
dst_width = 1472

def scale_to_normal(img):
    h, w, _ = img.shape
    if w < dst_width:
        print('[W] input image has size(w={}) smaller than normailize size(w={})'.format(w, dst_width))

    factor = dst_width / w
    return cv2.resize(img, (0, 0), fx=factor, fy=factor), factor


def adp_thresh_grayscale(gray, thr=250):
    # cv2.imshow('before equalizeHist', gray)
    img = cv2.equalizeHist(gray)
    # cv2.imshow('after equalizeHist', img)
    ret, thrs = cv2.threshold(img, thresh=thr, maxval=255, type=cv2.THRESH_BINARY)

    return thrs


def find_circle(img,
                valMedianBlur=5,
                valKernelOpen=5,
                valKernelClose=63,
                valAdaptivateThreshold=250,
                valHoughParam1=172,
                valHoughParam2=6,
                valHoughMinDist=900,
                show_debug_preview=True
                ):
    target_windowsname = 'final'
    img = img.copy()

    img, factor = scale_to_normal(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray_img = adp_thresh_grayscale(gray_img, valAdaptivateThreshold)

    # MORPH_OPEN is erosion followed by dilation,
    #   eliminate noise outside the target
    mopen_kernel = np.ones((valKernelOpen, valKernelOpen), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, mopen_kernel)
    if show_debug_preview: cv2.imshow('MORPH_OPEN', gray_img)

    # Eliminate the noise inside
    mclose_kernel = np.ones((valKernelClose, valKernelClose), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, mclose_kernel)
    if show_debug_preview: cv2.imshow('MORPH_CLOSE', gray_img)

    pimg = cv2.medianBlur(gray_img, valMedianBlur)
    if show_debug_preview: cv2.imshow('medianBlur', pimg)
    # pimg = cv2.cvtColor(pimg, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(pimg, cv2.HOUGH_GRADIENT, 1, valHoughMinDist,
                               param1=valHoughParam1, param2=valHoughParam2, minRadius=0, maxRadius=0)
    if circles is None:
        print('circles={}, skip.'.format(circles))
        cv2.destroyWindow(target_windowsname)
        return

    draw_circles = np.uint16(np.around(circles))
    ret = []

    for i in draw_circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)

    for i in circles[0, :]:

        # Scale the circle back to ori size
        x = int(i[0] / factor)
        y = int(i[1] / factor)
        z = int(i[2] / factor)

        ret.append(Circle(x, y, z))

    # cv2.imwrite("1.jpg", img)
    if show_debug_preview: cv2.imshow(target_windowsname, img)
    return ret


def loadimg():
    ori_img = cv2.imread(
        'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
    print('origin img.shape={}'.format(ori_img.shape))

    myTuneWindow(find_circle, ori_img,
                 valMedianBlur=(1, 30, 2),
                 valKernelOpen=(1, 100, 2),
                 valKernelClose=(1, 100, 2),
                 valAdaptivateThreshold=(1, 255),
                 valHoughParam1=(1, 300),
                 valHoughParam2=(1, 300),
                 valHoughMinDist=(1, 1000),
                 )


if __name__ == '__main__':
    loadimg()
