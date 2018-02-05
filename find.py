import logging

import cv2
import numpy as np

from utils.CVPipeline import CVPipeline
from utils.util import resize

logger = logging.getLogger(__name__)


class Circle():
    def __init__(self, x, y, r):
        self.r = r
        self.y = y
        self.x = x


# Normalize image
NORM_WIDTH = 500


def scale_to_normal(img):
    h, w, _ = img.shape
    if w < NORM_WIDTH:
        logger.warning('input image has size(w={}) smaller than normailize size(w={})'.format(w, NORM_WIDTH))

    factor = NORM_WIDTH / w
    if factor < 0.5:
        logger.warning('The ratio is too small (%.2f), may not produce the correct detection. (is the moon too small?)',
                       factor)
    return cv2.resize(img, (0, 0), fx=factor, fy=factor), factor


def adp_thresh_bin(gray, thr=250):
    # cv2.imshow('before equalizeHist', gray)
    img = cv2.equalizeHist(gray)
    # cv2.imshow('after equalizeHist', img)
    ret, thrs = cv2.threshold(img, thresh=thr, maxval=255, type=cv2.THRESH_BINARY)

    return thrs


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def find_circle(img,
                valMedianBlur=19,
                valKernelOpen=3,
                valKernelClose=85,
                valHoughParam1=47,
                valHoughParam2=6,
                valHoughMinDist=900,
                valBlfD=15,
                valBlfColor=750,
                valBlfSpace=750,
                valAdaptiveThreshold=141,
                show_debug_preview=True
                ):
    win_HoughCircles = 'HoughCircles'
    img = img.copy()

    img, factor = scale_to_normal(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray_img = adp_thresh_grayscale(gray_img, valAdaptivateThreshold)

    gray_img = cv2.bilateralFilter(gray_img, valBlfD, valBlfColor, valBlfSpace)
    if show_debug_preview: cv2.imshow('find_circle.bilateralFilter', gray_img)

    # Padding the image a little
    padding = int(NORM_WIDTH / 10)
    gray_img = cv2.copyMakeBorder(gray_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = padding_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_img = clahe.apply(gray_img)
    if show_debug_preview: cv2.imshow('find_circle.clahe', hist_img)

    blur_img = cv2.medianBlur(gray_img, valMedianBlur)
    if show_debug_preview: cv2.imshow('find_circle. medianBlur', blur_img)

    gray_img = adp_thresh_bin(gray_img, valAdaptiveThreshold)
    if show_debug_preview: cv2.imshow('find_circle.adp_thresh_bin', gray_img)

    # MORPH_OPEN is erosion followed by dilation,
    #   eliminate noise outside the target
    mopen_kernel = np.ones((valKernelOpen, valKernelOpen), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, mopen_kernel)
    if show_debug_preview: cv2.imshow('find_circle.MORPH_OPEN', gray_img)

    # Eliminate the noise inside
    mclose_kernel = np.ones((valKernelClose, valKernelClose), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, mclose_kernel)
    if show_debug_preview: cv2.imshow('find_circle.MORPH_CLOSE', gray_img)

    gray_img = cv2.addWeighted(gray_img, 0.05, blur_img, 1, 0)
    if show_debug_preview: cv2.imshow('find_circle.addWeighted', gray_img)

    # gray_img = auto_canny(gray_img)
    # if show_debug_preview: cv2.imshow('find_circle.auto_canny', gray_img)

    circles = cv2.HoughCircles(gray_img.copy(), cv2.HOUGH_GRADIENT, 1, valHoughMinDist,
                               param1=valHoughParam1, param2=valHoughParam2,
                               minRadius=int(NORM_WIDTH / 2 - NORM_WIDTH * 0.3),
                               maxRadius=int(NORM_WIDTH / 2 + NORM_WIDTH * 0.1))

    # --------------- Try using contour fit ----------------
    # gray_img = adp_thresh_bin(gray_img, valAdaptiveThreshold)
    # if show_debug_preview: cv2.imshow('find_circle.adp_thresh_bin', gray_img)

    # gray_img = cv2.bitwise_not(gray_img)
    # if show_debug_preview: cv2.imshow('find_circle.invert', gray_img)
    #
    # largest_contour = find_largest_contour(gray_img)
    #
    # if show_debug_preview:
    #     img_1 = img.copy()
    #     cv2.drawContours(img_1, largest_contour, -1, (0, 255, 0), 3)
    #     cv2.imshow('find_circle.drawContours', img_1)
    #
    #     # img_2 = img.copy()
    #     # epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    #     # approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    #     # cv2.drawContours(img_2, [approx], -1, (0, 255, 255), 3)
    #     # cv2.imshow('find_circle.drawContours(approxPolyDP)', img_2)
    #
    #     ellipse = cv2.fitEllipse(largest_contour)
    #     img_ellipse = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 2)
    #     cv2.imshow('find_circle.fitEllipse', img_ellipse)
    #     print("ellipse: %s" % str(ellipse))
    #
    #     # ellipse = cv2.fitEllipse(approx)
    #     # img_ellipse = cv2.ellipse(img.copy(), ellipse, (0, 255, 255), 2)
    #     # cv2.imshow('find_circle.fitEllipse(approxPolyDP)', img_ellipse)
    #     # print("ellipse: %s(approxPolyDP)" % str(ellipse))

    # Post processing -------------------------------------------------------------
    if circles is None:
        logger.warning('No circle found')
        cv2.destroyWindow(win_HoughCircles)
        return []
    draw_circles = np.uint16(np.around(circles))
    for i in draw_circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)

    ret = []
    for i in circles[0, :]:
        # Scale the circle back to ori size
        x = int((i[0] - padding) / factor)
        y = int((i[1] - padding) / factor)
        r = int((i[2]) / factor)

        ret.append(Circle(x, y, r))

    # cv2.imwrite("1.jpg", img)
    if show_debug_preview: cv2.imshow(win_HoughCircles, img)
    return ret


def find_largest_contour(bin_img):
    image, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0
    if len(contours) > 0:
        logger.debug('printing area of contours in `find_largest_contour()`')
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        logger.debug('\t[%d/%d] area==%s', i + 1, len(contours), area)
        if area > largest_area:
            largest_contour = c
            largest_area = area

    return largest_contour


def find_black_drop(img, remaining_percentage=0.5) -> int:
    """Use histogram to determine the major black threshold"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # hist is a 256x1 array
    #   [Black -> White]
    #   [  0   ..  255 ]

    # print(hist[:])

    def diff(list_):
        return [list_[n] - list_[n - 1] for n in range(1, len(list_))]

    s = np.sum(hist[:])
    hist = hist / s
    dhist = diff(hist)

    # print(hist)

    main_drop = min(dhist)
    remaining_thr = main_drop * remaining_percentage
    for i in range(dhist.index(main_drop), len(dhist)):
        if dhist[i] < remaining_thr:
            logger.debug('\tmoving to index=%d and checking remaining', i + 1)
            continue
        else:
            return i
    logger.warning('reach the end of histogram, not found remaining threshold. min=%f, minindex=%d', main_drop,
                   dhist.index(main_drop))
    return dhist.index(main_drop)

    # for i, h in enumerate(hist):
    #     if i + tolerance >= len(hist):
    #         # todo maybe not found
    #         return i
    #     if h - hist[i+tolerance] > abs(delta):
    #         # return the first found
    #         return int(i + tolerance/2)


class FindMainObject(CVPipeline):
    def _pipeline(self, *inputargs):
        img = inputargs[0]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find a suitable threshold from histogram
        def auto_threshold(img, _percent=0.05):
            majorblack = find_black_drop(img, _percent)
            logger.debug('majorblack==%d', majorblack)
            _, gray_img = cv2.threshold(img, majorblack, 255, cv2.THRESH_BINARY)
            return gray_img
        img_auto_thr = self._add_tune_step(auto_threshold, img_gray, _percent=(0.001, 0.6))

        # For comparision
        def threshold_bin(img, _threshold=10):
            ret, retimg = cv2.threshold(img, _threshold, 255, cv2.THRESH_BINARY)
            return retimg
        self._add_tune_step(threshold_bin, img_gray, _threshold=(0, 255))

        # MORPH_OPEN is erosion followed by dilation,
        #   eliminate noise outside the target
        def morphopen(img, _kernel=43):
            mopen_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, mopen_kernel)
        img_ = self._add_tune_step(morphopen, img_auto_thr, _kernel=(1, 100, 2))

        def morphclose(img, _kernel=73):
            mclose_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, mclose_kernel)
        img_ = self._add_tune_step(morphclose, img_, _kernel=(1, 100, 2))

        def mediumblur(img, _blur=5):
            return cv2.medianBlur(img, _blur)
        img_ = self._add_tune_step(mediumblur, img_, _blur=(1, 30, 2))

        _, contours, hierarchy = cv2.findContours(img_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = find_largest_contour(img_)
        x_, y_, w_, h_ = cv2.boundingRect(largest_contour)
        if w_ == 0 or h_ == 0:
            self.logger.warning('find boundingRect failed: w=%d, h=%d', w_, h_)
            return None, None, None

        roi = img[y_:y_ + h_, x_:x_ + w_]
        self._add_debug_view('roi', roi)

        logger.debug('main_object roi x={}, y={}, w={}, h={}'.format(x_, y_, w_, h_))
        return roi, x_, y_


def find_main_object(img,
                     _kernelOpen=43,
                     _kernelClose=73,
                     _medianBlur=5,
                     _threshold=10,
                     ):
    """Find the main object (a circle) in img"""
    previewwin_w = 500


    masked = img
    gray_img = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # # create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray_img = clahe.apply(gray_img)
    # if show_debug_preview: cv2.imshow('find_main_circle.clahe', resize_bin(gray_img, previewwin_w))
    gray_img_ = gray_img.copy()

    # gray_img = adp_thresh_bin(gray_img, valAdaptiveThreshold1)
    # if show_debug_preview: cv2.imshow('find_main_circle.adp_thresh_bin1', resize_bin(gray_img, previewwin_w))

    # Find a suitable threshold from histogram
    majorblack = find_black_drop(gray_img, 0.05)
    logger.debug('majorblack==%d', majorblack)
    ret, gray_img = cv2.threshold(gray_img, majorblack, 255, cv2.THRESH_BINARY)
    if show_debug_preview: cv2.imshow('find_main_circle.majorblack', resize(gray_img, previewwin_w))

    ret, test_view = cv2.threshold(gray_img_, _threshold, 255, cv2.THRESH_BINARY)
    if show_debug_preview: cv2.imshow('find_main_circle.valThr'.format(_threshold), resize(test_view, previewwin_w))

    # gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C ,
    #                                  cv2.THRESH_BINARY, blockSize=valAdpBSize, C=valAdpC)
    # if show_debug_preview:
    #     cv2.imshow('find_main_circle.adaptiveThreshold', resize_bin(gray_img, previewwin_w))

    # MORPH_OPEN is erosion followed by dilation,
    #   eliminate noise outside the target
    mopen_kernel = np.ones((_kernelOpen, _kernelOpen), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, mopen_kernel)
    if show_debug_preview:
        cv2.imshow('find_main_circle.MORPH_OPEN', resize(gray_img, previewwin_w))

    # Eliminate the noise inside
    mclose_kernel = np.ones((_kernelClose, _kernelClose), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, mclose_kernel)
    if show_debug_preview:
        cv2.imshow('find_main_circle.MORPH_CLOSE', resize(gray_img, previewwin_w))

    pimg = cv2.medianBlur(gray_img, _medianBlur)
    if show_debug_preview:
        cv2.imshow('find_main_circle.medianBlur', resize(pimg, previewwin_w))

    # gray_img = adp_thresh_bin(pimg, valAdaptiveThreshold2)
    # if show_debug_preview:
    #     cv2.imshow('find_main_circle.adp_thresh_bin2', resize_bin(gray_img, previewwin_w))

    image, contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = find_largest_contour(gray_img)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w == 0 or h == 0:
        logger.warning('in find_main_object(), find boundingRect failed, w=%d, h=%d', w, h)
        return None, None, None

    if show_debug_preview:
        img_1 = img.copy()
        cv2.drawContours(img_1, contours, -1, (0, 255, 0), 3)
        cv2.imshow('find_main_object.drawContours', resize(img_1, previewwin_w * 2))

        try:
            img_2 = img.copy()
            ellipse = cv2.fitEllipse(largest_contour)
            img_2 = cv2.ellipse(img_2, ellipse, (0, 255, 0), 3)
            cv2.imshow('find_main_object.fitEllipse', resize(img_2, previewwin_w * 2))
        except:
            cv2.destroyWindow('find_main_object.fitEllipse')
            logger.exception('failed to fitEllipse')

        # img_3 = img.copy()
        # img_3 = cv2.rectangle(img_3, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('findContours.boundingRect', resize(img_3, previewwin_w*2))

    roi = img[y:y + h, x:x + w]
    if show_debug_preview:
        cv2.imshow('roi', roi)

    logger.debug('main_object roi x={}, y={}, w={}, h={}'.format(x, y, w, h))

    return roi, x, y


def draw_moon(img, moon: Circle):
    cv2.circle(img, (moon.x, moon.y), moon.r, (0, 255, 0), 2)
    # draw the center of the moon
    cv2.circle(img, (moon.x, moon.y), 5, (0, 0, 255), 1)


def find_moon(img) -> Circle:
    roi, rx, ry = find_main_object(img, show_debug_preview=False)
    if roi is None:
        logger.error('Failed to locate a main object, aborted.')
        return None
    circles = find_circle(roi, show_debug_preview=False)

    if not circles:
        return None

    ret_circle = circles[0]
    if len(circles) > 1:
        logger.warning('Warning: detected more than one circle, use the largest one')

        largest_r = 0
        for c in circles:
            if c.r > largest_r:
                ret_circle = c
                largest_r = c.r

    # Offset the roi
    ret_circle.x += rx
    ret_circle.y += ry

    return ret_circle
