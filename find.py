import logging

import cv2
import numpy as np

from utils.CVPipeline import CVPipeline
from utils.ContourSelector import ContourSelector

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

    @staticmethod
    def fit_circle(arc_contour):
        """
        Fit the circle with feeding points

        Algorithm from scipy-cookbook: http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        :param arc_contour:
        :return: (x, y, r)
        """
        from scipy import odr

        x = np.array([i[0][0] for i in arc_contour])
        y = np.array([i[0][1] for i in arc_contour])

        def calc_R(c):
            """ calculate the distance of each 2D points from the center c=(xc, yc) """
            return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)

        def circlemodel(beta, x):
            """ implicit function of the circle """
            xc, yc, r = beta
            return (x[0] - xc) ** 2 + (x[1] - yc) ** 2 - r ** 2

        def calc_estimate(data):
            """ Return a first estimation on the parameter from the data  """
            xc0, yc0 = data.x.mean(axis=1)
            r0 = np.sqrt((data.x[0] - xc0) ** 2 + (data.x[1] - yc0) ** 2).mean()
            return xc0, yc0, r0

        # for implicit function :
        #       data.x contains both coordinates of the points
        #       data.y is the dimensionality of the response
        lsc_data = odr.Data(np.row_stack([x, y]), y=1)
        lsc_model = odr.Model(circlemodel, implicit=True, estimate=calc_estimate)
        lsc_odr = odr.ODR(lsc_data, lsc_model)
        lsc_out = lsc_odr.run()

        xc_3, yc_3, R_3 = lsc_out.beta
        print('lsc_out.sum_square = ', lsc_out.sum_square)

        # Ri_3 = calc_R([xc_3, yc_3])
        # residu_3 = sum((Ri_3 - R_3) ** 2)
        # residu2_3 = sum((Ri_3 ** 2 - R_3 ** 2) ** 2)
        # print('residu_3  :', residu_3 )
        # print('residu2_3 :', residu2_3)

        return xc_3, yc_3, R_3

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
        inputimg = inputargs[0]

        img_scaled, factor = self.scale_to_normal(inputimg)
        img_ = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

        def biltfilter(img, _d=15, _sigmacolor=750, _sigmaspace=750):
            return cv2.bilateralFilter(img, _d, _sigmacolor, _sigmaspace)
        img_ = self._add_tune_step(biltfilter, img_, _d=(0, 30), _sigmacolor=(0, 1250), _sigmaspace=(0, 1250))

        def medianblur(img, _blur=19):
            return cv2.medianBlur(img, _blur)
        img_ = img_blur = self._add_tune_step(medianblur, img_, _blur=(1, 30, 2))

        def canny(img, _sigma=0.33):
            return auto_canny(img, sigma=_sigma)
        img_ = self._add_tune_step(canny, img_, _sigma=(0.01, 50.0))

        # Create a mask to eliminate the detail of the moon surface
        _, img_mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # self._add_debug_view('otsu threshold', img_otsu)
        # Closing is Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects
        def morphclose(img, _kernel=85):
            mclose_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, mclose_kernel)
        img_mask = self._add_tune_step(morphclose, img_mask, _kernel=(1, 100, 2))
        # Use erosion to make the mask region smaller
        def erosionimg(img, _kernel=3):
            k = np.ones((_kernel, _kernel), np.uint8)
            return cv2.erode(img, k, iterations=1)
        img_mask = self._add_tune_step(erosionimg, img_mask, _kernel=(1, 100, 2))

        # Mask the canny edge, erase moon surface detail
        img_ = img_ & cv2.bitwise_not(img_mask)
        self._add_debug_view('masked canny', img_)

        # Extends the mask a little bit
        def dilationmask(img, _kernel=3):
            k = np.ones((_kernel, _kernel), np.uint8)
            return cv2.dilate(img, k, iterations=1)
        img_mask = self._add_tune_step(dilationmask, img_mask, _kernel=(1, 100, 2))

        # Mask the canny edge again, hope to eliminate the edge form by shaded part (the sharp edge will be kept)
        img_ = img_ & (img_mask)
        self._add_debug_view('masked canny 2', img_)

        if False:  # Use padding
            padding = int(self.NORM_WIDTH / 20)
            img_ = img_gray = cv2.copyMakeBorder(img_, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img_scaled = cv2.copyMakeBorder(img_scaled, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            padding = 0
            img_gray = img_.copy()

        # Extend the canny edge a little bit
        def dilationimg(img, _kernel=3):
            k = np.ones((_kernel, _kernel), np.uint8)
            return cv2.dilate(img, k, iterations=1)
        img_ = self._add_tune_step(dilationimg, img_, _kernel=(1, 100, 2))

        # MORPH_OPEN is erosion followed by dilation, eliminate noise outside the target
        def morphopen(img, _kernel=3):
            mopen_kernel = np.ones((_kernel, _kernel), np.uint8)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, mopen_kernel)
        img_ = self._add_tune_step(morphopen, img_, _kernel=(1, 100, 2), show_preview=False)

        # # Eliminate the noise inside
        # def morphclose(img, _kernel=85):
        #     mclose_kernel = np.ones((_kernel, _kernel), np.uint8)
        #     return cv2.morphologyEx(img, cv2.MORPH_CLOSE, mclose_kernel)
        # img_ = self._add_tune_step(morphclose, img_, _kernel=(1, 100, 2))

        def medianblur2(img, _blur=19):
            return cv2.medianBlur(img, _blur)
        img_ = self._add_tune_step(medianblur2, img_, _blur=(1, 30, 2), show_preview=False)

        # -----------------------------------------------------------------------------------------------------------
        # Try using contour method

        largest_contour = find_largest_contour(img_)
        if largest_contour is None:
            self.logger.warning('no contour found.')
            return None

        self.logger.debug('contour has {} points'.format(len(largest_contour)))
        img_1 = img_scaled.copy()
        cv2.drawContours(img_1, largest_contour, -1, (0, 255, 0), 2)
        self._add_debug_view('drawContours', img_1)

        if False:  # Select some contour manually
            c = ContourSelector(img_scaled.copy(), largest_contour)
            c.show_and_interact()
            arc_contour = c.get_selected()
            cx, cy, cr = self.fit_circle(arc_contour)
        else:
            cx, cy, cr = self.fit_circle(largest_contour)

        # Remove padding and scale back
        _x = int((cx - padding) / factor)
        _y = int((cy - padding) / factor)
        _r = int(cr / factor)
        ret_circle = Circle(_x, _y, _r)

        # Draw the fitting circle for preview
        (cx, cy, cr) = np.uint16(np.around((cx, cy, cr)))
        img_2 = img_scaled.copy()
        cv2.circle(img_2, (cx, cy), cr, (0, 255, 0), 1)
        cv2.circle(img_2, (cx, cy), 2, (0, 0, 255), 1)
        self._add_debug_view('fitcircle', img_2)

        img_3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_3, (cx, cy), cr, (0, 255, 0), 1)
        cv2.circle(img_3, (cx, cy), 2, (0, 0, 255), 1)
        self._add_debug_view('fitcircle vs canny', img_3)

        # Scale the circle back to ori size
        return ret_circle

        # -----------------------------------------------------------------------------------------------------------
        # img_ = cv2.addWeighted(img_gray, 0.2, img_, 0.8, 0)
        # # self._add_debug_view('addWeighted', img_)
        #
        # def h_circle(img, img_ori, _min_dist=900, _param1=47, _param2=6):
        #     circles = cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, 1,
        #                                minDist=_min_dist, param1=_param1, param2=_param2,
        #                                minRadius=int(self.NORM_WIDTH / 2 - self.NORM_WIDTH * 0.3),
        #                                maxRadius=int(self.NORM_WIDTH / 2 + self.NORM_WIDTH * 0.1))
        #     if circles is None:
        #         self.logger.warning('No circle found')
        #         return None, None
        #     draw_circles = np.uint16(np.around(circles))
        #     img_draw = img_ori.copy()
        #     for i in draw_circles[0, :]:
        #         # Draw the outer circle
        #         cv2.circle(img_draw, (i[0], i[1]), i[2], (0, 255, 0), 1)
        #         # Draw the center of the circle
        #         cv2.circle(img_draw, (i[0], i[1]), 2, (0, 0, 255), 1)
        #     return img_draw, circles
        # _, circles = self._add_tune_step(h_circle, img_, img_scaled,
        #                                  _min_dist=(1, 1000), _param1=(1, 300), _param2=(1, 300), show_preview=False)
        #
        # ret = []
        # if circles is None:
        #     return []
        # for i in circles[0, :]:
        #     # Scale the circle back to ori size
        #     x = int((i[0] - padding) / factor)
        #     y = int((i[1] - padding) / factor)
        #     r = int((i[2]) / factor)
        #
        #     ret.append(Circle(x, y, r))
        # return ret


def find_largest_contour(bin_img):
    """Find and return the contour that has the largest area"""
    image, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0
    if len(contours) > 0:
        # logger.debug('printing area of contours in `find_largest_contour()`')
        pass
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        # logger.debug('\t[%d/%d] area==%s', i + 1, len(contours), area)
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


def padding_crop(bigimg, padding, x, y, w, h):
    """Crop the image to a roi, with some padding
    Return the crop image and the crop rectangle
    """
    iw, ih = bigimg.shape[1], bigimg.shape[0]
    # Padding if possible
    left, right = np.uint16(np.around((max(x - padding, 0), min(iw, x + w + padding))))
    top, bottom = np.uint16(np.around((max(y - padding, 0), min(ih, y + h + padding))))

    return bigimg[top:bottom, left: right], (left, top, right-left, bottom-top)


class FindMainObject(CVPipeline):
    def _pipeline(self, *inputargs):
        img = inputargs[0]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # For comparision
        # # Find a suitable threshold from histogram
        # def auto_threshold(img, _percent=0.05):
        #     majorblack = find_black_drop(img, _percent)
        #     self.logger.debug('majorblack==%d', majorblack)
        #     _, gray_img = cv2.threshold(img, majorblack, 255, cv2.THRESH_BINARY)
        #     return gray_img
        # self._add_tune_step(auto_threshold, img_gray, _percent=(0.001, 0.6))

        def automain(img, _iteration=3, _kernelclose=17, _kernelopen=3):
            global gx, gy, gw, gh
            gx, gy, gw, gh = 0, 0, 0, 0

            def run_(img):
                thr, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                mopen_kernel = np.ones((_kernelopen, _kernelopen), np.uint8)
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, mopen_kernel)

                mclose_kernel = np.ones((_kernelclose, _kernelclose), np.uint8)
                bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, mclose_kernel)

                _, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = find_largest_contour(bin_img)
                x_, y_, w_, h_ = cv2.boundingRect(largest_contour)
                if w_ == 0 or h_ == 0:
                    return None

                # Padding if possible
                p = (h_ + w_) / 2 * 0.1  # 10% of avg
                img_crop, (rx, ry, rw, rh) = padding_crop(img, p, x_, y_, w_, h_)

                self.logger.debug('cropping to x=%d y=%d w=%d h=%d, current threshold=%d', rx, ry, rw, rh, thr)

                global gx, gy, gw, gh
                gx, gy, gw, gh = gx+rx, gy+ry, rw, rh

                return img_crop

            for i in range(_iteration):
                last_img = img
                img = run_(img)
                if img is None:
                    self.logger.debug('_iteration break at %d', i)
                    return last_img, (gx, gy, gw, gh)

            _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bin_img, (gx, gy, gw, gh)

        img_, (mx, my, mw, mh) = self._add_tune_step(automain, img_gray,
                                              _iteration=(0, 10), _kernelclose=(1, 100, 2), _kernelopen=(1, 100, 2))
        if mw == 0 or mh == 0:
            self.logger.warning('failed in automain')
            return None, (0, 0)

        # # For comparision
        # def threshold_bin(img, _threshold=45):
        #     ret, retimg = cv2.threshold(img, _threshold, 255, cv2.THRESH_BINARY)
        #     return retimg
        # self._add_tune_step(threshold_bin, img_, _threshold=(0, 255))

        # # For comparision
        # def adpthreshold(img, _blocksize=11, _c=2):
        #     return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                  cv2.THRESH_BINARY, _blocksize, _c)
        # self._add_tune_step(adpthreshold, img_gray, _blocksize=(1, 1000, 2), _c=(0, 50))

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

        # Get location
        _, contours, hierarchy = cv2.findContours(img_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = find_largest_contour(img_)
        x_, y_, w_, h_ = cv2.boundingRect(largest_contour)
        if w_ == 0 or h_ == 0:
            self.logger.warning('failed to find boundingRect: w=%d, h=%d', w_, h_)
            return None, (0, 0)

        # Offset the 1st crop
        x_ += mx
        y_ += my

        # Padding and crop the roi
        padding = (w_ + h_)/2 * 0.1  # padding %10 of avg
        img_roi, (x, y, w, h) = padding_crop(img, padding, x_, y_, w_, h_)
        self._add_debug_view('main object roi', img_roi)

        self.logger.debug('main_object roi x={}, y={}, w={}, h={}'.format(x, y, w, h))
        return img_roi, (x, y)


def draw_moon(img, moon: Circle):
    cv2.circle(img, (moon.x, moon.y), moon.r, (0, 255, 0), 2)
    # draw the center of the moon
    cv2.circle(img, (moon.x, moon.y), 5, (255, 0, 255), 1)


def find_moon(img, f0: FindMainObject, f1: FindCircle, tune=False) -> Circle:
    p0 = f0.run_pipeline_final
    p1 = f1.run_pipeline_final
    if tune:
        p0 = f0.run_pipeline_tuning
        p1 = f1.run_pipeline_tuning

    roi, (rx, ry) = p0(img)
    if roi is None:
        logger.error('Failed to locate a main object, aborted.')
        return None

    circle_ = p1(roi)
    if not circle_:
        return None

    # Offset the roi
    circle_.x += rx
    circle_.y += ry

    return circle_
