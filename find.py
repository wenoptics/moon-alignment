import cv2
import numpy as np


class Circle():
    def __init__(self, x, y, r):
        self.r = r
        self.y = y
        self.x = x


# Normalize image
DST_WIDTH = 500


def resize(img, dst_width):
    h, w, _ = img.shape
    factor = dst_width / w
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)


def resize_bin(img, dst_width):
    h, w = img.shape
    factor = dst_width / w
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)


def scale_to_normal(img):
    h, w, _ = img.shape
    if w < DST_WIDTH:
        print('[W] input image has size(w={}) smaller than normailize size(w={})'.format(w, DST_WIDTH))

    factor = DST_WIDTH / w
    if factor < 0.5:
        print('[W] The ratio is too small ({}), may not produce the correct detection. (is the moon too small?)'.format(factor))
    return cv2.resize(img, (0, 0), fx=factor, fy=factor), factor


def adp_thresh_bin(gray, thr=250):
    # cv2.imshow('before equalizeHist', gray)
    img = cv2.equalizeHist(gray)
    # cv2.imshow('after equalizeHist', img)
    ret, thrs = cv2.threshold(img, thresh=thr, maxval=255, type=cv2.THRESH_BINARY)

    return thrs


def find_circle(img,
                valMedianBlur=19,
                valKernelOpen=5,
                valKernelClose=7,
                valHoughParam1=172,
                valHoughParam2=6,
                valHoughMinDist=900,
                valAdaptiveThreshold=107,
                show_debug_preview=True
                ):
    win_HoughCircles = 'HoughCircles'
    img = img.copy()

    img, factor = scale_to_normal(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray_img = adp_thresh_grayscale(gray_img, valAdaptivateThreshold)

    # MORPH_OPEN is erosion followed by dilation,
    #   eliminate noise outside the target
    mopen_kernel = np.ones((valKernelOpen, valKernelOpen), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, mopen_kernel)
    if show_debug_preview: cv2.imshow('find_circle.MORPH_OPEN', gray_img)

    # Eliminate the noise inside
    mclose_kernel = np.ones((valKernelClose, valKernelClose), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, mclose_kernel)
    if show_debug_preview: cv2.imshow('find_circle.MORPH_CLOSE', gray_img)

    pimg = cv2.medianBlur(gray_img, valMedianBlur)
    if show_debug_preview: cv2.imshow('find_circle.medianBlur', pimg)
    # pimg = cv2.cvtColor(pimg, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(pimg.copy(), cv2.HOUGH_GRADIENT, 1, valHoughMinDist,
                               param1=valHoughParam1, param2=valHoughParam2, minRadius=0, maxRadius=0)
    if circles is None:
        print('circles={}, skip.'.format(circles))
        cv2.destroyWindow(win_HoughCircles)
        return

    # --------------- Try using contour fit ----------------
    gray_img = adp_thresh_bin(pimg, valAdaptiveThreshold)
    if show_debug_preview: cv2.imshow('find_circle.adp_thresh_bin', gray_img)

    largest_contour = find_largest_contour(gray_img)
    ellipse = cv2.fitEllipse(largest_contour)
    img_ellipse = cv2.ellipse(img.copy(), ellipse, (0, 255, 0), 2)
    if show_debug_preview: cv2.imshow('fitEllipse', img_ellipse)

    ret = []
    draw_circles = np.uint16(np.around(circles))
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
    if show_debug_preview: cv2.imshow(win_HoughCircles, img)
    return ret


def find_largest_contour(bin_img):
    image, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        print('\tarea==%s' % area)
        if area > largest_area:
            largest_contour = c
            largest_area = area

    return largest_contour


def find_main_object(img,
                     valKernelOpen=5,
                     valKernelClose=63,
                     valAdaptiveThreshold1=250,
                     valAdaptiveThreshold2=250,
                     valMedianBlur=5,
                     show_debug_preview=True):
    win_w = 500

    # Find the main object (a circle) in img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = adp_thresh_bin(gray_img, valAdaptiveThreshold1)
    if show_debug_preview: cv2.imshow('find_main_circle.adp_thresh_bin1', resize_bin(gray_img, win_w))

    # MORPH_OPEN is erosion followed by dilation,
    #   eliminate noise outside the target
    mopen_kernel = np.ones((valKernelOpen, valKernelOpen), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, mopen_kernel)
    if show_debug_preview:
        cv2.imshow('find_main_circle.MORPH_OPEN', resize_bin(gray_img, win_w))

    # Eliminate the noise inside
    mclose_kernel = np.ones((valKernelClose, valKernelClose), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, mclose_kernel)
    if show_debug_preview:
        cv2.imshow('find_main_circle.MORPH_CLOSE', resize_bin(gray_img, win_w))

    pimg = cv2.medianBlur(gray_img, valMedianBlur)
    if show_debug_preview:
        cv2.imshow('find_main_circle.medianBlur', resize_bin(pimg, win_w))

    gray_img = adp_thresh_bin(pimg, valAdaptiveThreshold2)
    if show_debug_preview:
        cv2.imshow('find_main_circle.adp_thresh_bin1', resize_bin(gray_img, win_w))

    image, contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = find_largest_contour(gray_img)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if show_debug_preview:
        img_1 = img.copy()
        cv2.drawContours(img_1, contours, -1, (0, 255, 0), 3)
        cv2.imshow('findContours.drawContours', resize(img_1, win_w*2))

        img_2 = img.copy()
        ellipse = cv2.fitEllipse(largest_contour)
        img_2 = cv2.ellipse(img_2, ellipse, (0, 255, 0), 3)
        cv2.imshow('findContours.fitEllipse', resize(img_2, win_w*2))

        # img_3 = img.copy()
        # img_3 = cv2.rectangle(img_3, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('findContours.boundingRect', resize(img_3, win_w*2))

    roi = img[y:y+h, x:x+w]
    if show_debug_preview:
        cv2.imshow('roi', roi)

    print('roi x={}, y={}, w={}, h={}'.format(x, y, w, h))

    return roi, x, y


def draw_moon(img, moon: Circle):
    cv2.circle(img, (moon.x, moon.y), moon.r, (0, 255, 0), 2)
    # draw the center of the moon
    cv2.circle(img, (moon.x, moon.y), 5, (0, 0, 255), 1)


def find_moon(img) -> Circle:
    roi, rx, ry = find_main_object(img, show_debug_preview=False)
    circles = find_circle(roi, show_debug_preview=False)

    if not circles:
        return None

    ret_circle = circles[0]
    if len(circles) > 1:
        print('Warning: detected more than one circle')

        largest_r = 0
        for c in circles:
            if c.r > largest_r:
                ret_circle = c
                largest_r = c.r

    # Offset the roi
    ret_circle.x += rx
    ret_circle.y += ry

    return ret_circle


