import logging

import cv2

from utils import util


class ContourSelector:
    def __init__(self, img, initcontour, name='contour selector'):
        self.name = name
        self.initcontour = initcontour
        self.selectedpoints = []
        self.img = img.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _draw_init_contour(self, img, color=(220,220,220), thickness=1):
        return cv2.drawContours(img, self.initcontour, -1, color=color, thickness=thickness)

    def _draw_selected_contour(self, img, color=(11,255,11), thickness=2):
        return cv2.drawContours(img, self.selectedpoints, -1, color=color, thickness=thickness)

    def _update_selected_contour(self, rect, mode):
        left = min(rect[0][0], rect[1][0])
        top = min(rect[0][1], rect[1][1])
        right = max(rect[0][0], rect[1][0])
        bottom = max(rect[0][1], rect[1][1])

        ndelete = 0
        tmplist = []
        for i in range(len(self.selectedpoints)):
            x = self.selectedpoints[i][0][0]
            y = self.selectedpoints[i][0][1]
            if x < left or x > right or y < top or y > bottom:
                ndelete += 1
            else:
                tmplist.append(self.selectedpoints[i])
        self.logger.info('excluded %d points', ndelete)
        self.selectedpoints = tmplist

    def show_and_interact(self):
        global ix, iy, drawing, img_tmp
        drawing = False  # true if mouse is pressed
        ix, iy = -1, -1

        def reset():
            global img_tmp
            img_tmp = self.img.copy()
            self._draw_init_contour(img_tmp)
            self._draw_selected_contour(img_tmp)

        self.selectedpoints = self.initcontour
        reset()

        # Mouse callback function
        def on_mouse(event, x, y, flags, param):
            global ix, iy, drawing

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    reset()
                    cv2.rectangle(img_tmp, (ix, iy), (x, y), (0, 255, 0), 1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                self._update_selected_contour(rect=((ix, iy), (x, y)))
                reset()
                # cv2.rectangle(img_tmp, (ix, iy), (x, y), (170, 255, 170), 1)

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, on_mouse)

        self.logger.info('Press "r" to reset; "q" to quit; "m" to change select mode')
        while True:
            cv2.imshow(self.name, img_tmp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                pass
            if k == ord('r'):
                self.selectedpoints = self.initcontour
                reset()
            if k == ord('q'):
                break

        cv2.destroyWindow(self.name)

    def get_selected(self):
        """Return selected contours"""
        return self.selectedpoints


if __name__ == '__main__':

    test_img = util.resize(cv2.imread('../test_dataset/DSC01332.jpg'), 500)
    c = ContourSelector(test_img, [])
    c.show_and_interact()