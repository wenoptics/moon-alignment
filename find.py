import logging

import cv2
import numpy as np

from utils.CVPipeline import CVPipeline

logger = logging.getLogger(__name__)


class Circle():
    def __init__(self, x, y, r):
        self.r = r
        self.y = y
        self.x = x


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


class FindCircle(CVPipeline):

    # Normalize image
    NORM_WIDTH = 500

    def scale_to_normal(self, img):
        h, w, _ = img.shape
        if w < self.NORM_WIDTH:
            self.logger.warning('input image has size(w={}) smaller than normailize size(w={})'.format(w, self.NORM_WIDTH))

        factor = self.NORM_WIDTH / w
        if factor < 0.5:
            self.logger.warning(
                'The ratio is too small (%.2f), may not produce the correct detection. (is the moon too small?)',
                factor)
        return cv2.resize(img, (0, 0), fx=factor, fy=factor), factor

    def _pipeline(self, *inputargs):
        img = inputargs[0]

        img_scaled, factor = self.scale_to_normal(img)
        img_ = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

        def biltfilter(img, _d=15, _sigmacolor=750, _sigmaspace=750):
            return cv2.bilateralFilter(img, _d, _sigmacolor, _sigmaspace)
        img_ = self._add_tune_step(biltfilter, img_, _d=(0, 30), _sigmacolor=(0, 1250), _sigmaspace=(0, 1250))

        padding = int(self.NORM_WIDTH / 20)
        img_ = img_gray = cv2.copyMakeBorder(img_, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img_scaled = cv2.copyMakeBorder(img_scaled, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        def medianblur(img, _blur=19):
            return cv2.medianBlur(img, _blur)
        img_ = self._add_tune_step(medianblur, img_, _blur=(1, 30, 2))

        _, img_otsu = cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self._add_debug_view('otsu threshold', img_otsu)

        # MORPH_OPEN is erosion followed by dilation,
        #   eliminate noise outside the target
        def morphopen(img, _kernel=3):
            mopen_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, mopen_kernel)
        img_ = self._add_tune_step(morphopen, img_otsu, _kernel=(1, 100, 2))

        # Eliminate the noise inside
        def morphclose(img, _kernel=85):
            mclose_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, mclose_kernel)
        img_ = self._add_tune_step(morphclose, img_, _kernel=(1, 100, 2))

        img_ = cv2.addWeighted(img_gray, 0.2, img_, 0.8, 0)
        self._add_debug_view('addWeighted', img_)

        def h_circle(img, img_ori, _min_dist=900, _param1=47, _param2=6):
            circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1,
                                       minDist=_min_dist, param1=_param1, param2=_param2,
                                       minRadius=int(self.NORM_WIDTH / 2 - self.NORM_WIDTH * 0.3),
                                       maxRadius=int(self.NORM_WIDTH / 2 + self.NORM_WIDTH * 0.1))
            if circles is None:
                self.logger.warning('No circle found')
                return None, None
            draw_circles = np.uint16(np.around(circles))
            img_draw = img_ori.copy()
            for i in draw_circles[0, :]:
                # Draw the outer circle
                cv2.circle(img_draw, (i[0], i[1]), i[2], (0, 255, 0), 1)
                # Draw the center of the circle
                cv2.circle(img_draw, (i[0], i[1]), 2, (0, 0, 255), 1)
            return img_draw, circles
        _, circles = self._add_tune_step(h_circle, img_, img_scaled, _min_dist=(1, 1000), _param1=(1, 300), _param2=(1, 300))

        ret = []
        if circles is None:
            return []
        for i in circles[0, :]:
            # Scale the circle back to ori size
            x = int((i[0] - padding) / factor)
            y = int((i[1] - padding) / factor)
            r = int((i[2]) / factor)

            ret.append(Circle(x, y, r))
        return ret

'''
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
'''


def find_largest_contour(bin_img):
    """Find and return the contour that has the largest area"""
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

        # For comparision
        # Find a suitable threshold from histogram
        def auto_threshold(img, _percent=0.05):
            majorblack = find_black_drop(img, _percent)
            self.logger.debug('majorblack==%d', majorblack)
            _, gray_img = cv2.threshold(img, majorblack, 255, cv2.THRESH_BINARY)
            return gray_img
        self._add_tune_step(auto_threshold, img_gray, _percent=(0.001, 0.6))

        # For comparision
        def threshold_bin(img, _threshold=10):
            ret, retimg = cv2.threshold(img, _threshold, 255, cv2.THRESH_BINARY)
            return retimg
        self._add_tune_step(threshold_bin, img_gray, _threshold=(0, 255))

        # Auto threshold using Otsu threshold
        tval, img_ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.logger.info('value using using Otsu threshold == %d', tval)
        self._add_debug_view('otsu threshold', img_)

        # MORPH_OPEN is erosion followed by dilation,
        #   eliminate noise outside the target
        def morphopen(img, _kernel=43):
            mopen_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, mopen_kernel)
        img_ = self._add_tune_step(morphopen, img_, _kernel=(1, 100, 2))

        # Eliminate the noise inside
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

        # img_1 = img.copy()
        # cv2.drawContours(img_1, contours, -1, (0, 255, 0), 3)
        # self._add_debug_view('find_main_object.drawContours', img_1[y_:y_ + h_, x_:x_ + w_])
        #
        # img_2 = img.copy()
        # ellipse = cv2.fitEllipse(largest_contour)
        # img_2 = cv2.ellipse(img_2, ellipse, (0, 255, 0), 3)
        # self._add_debug_view('find_main_object.fitEllipse', img_2[y_:y_ + h_, x_:x_ + w_])

        roi = img[y_:y_ + h_, x_:x_ + w_]
        self._add_debug_view('roi', roi)

        self.logger.debug('main_object roi x={}, y={}, w={}, h={}'.format(x_, y_, w_, h_))
        return roi, x_, y_


def draw_moon(img, moon: Circle):
    cv2.circle(img, (moon.x, moon.y), moon.r, (0, 255, 0), 2)
    # draw the center of the moon
    cv2.circle(img, (moon.x, moon.y), 5, (0, 0, 255), 1)


def find_moon(img, f0: FindMainObject, f1: FindCircle, tune=False) -> Circle:

    p0 = f0.run_pipeline_final
    p1 = f1.run_pipeline_final
    if tune:
        p0 = f0.run_pipeline_tuning
        p1 = f1.run_pipeline_tuning

    roi, rx, ry = p0(img)
    if roi is None:
        logger.error('Failed to locate a main object, aborted.')
        return None

    circles = p1(roi)
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
