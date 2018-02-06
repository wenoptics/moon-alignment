import logging

import cv2

from utils import util


class ContourSelector:
    def __init__(self, img, initcontour, name='contour selector'):
        self.name = name
        self.initcontour = initcontour
        self.selected_tf_table = [False]*len(self.initcontour)
        self.img = img.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _draw_mode_indicator(self, img, mode):
        return cv2.putText(img, 'selection mode: {}'.format(mode), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (11,255,11))

    def _draw_init_contour(self, img, color=(220,220,220), thickness=1):
        return cv2.drawContours(img, self.initcontour, -1, color=color, thickness=thickness)

    def _draw_selected_contour(self, img, color=(11,255,11), thickness=2):
        selected = self.get_selected()
        return cv2.drawContours(img, selected, -1, color=color, thickness=thickness)

    def _update_selected_contour(self, rect, mode):
        left = min(rect[0][0], rect[1][0])
        top = min(rect[0][1], rect[1][1])
        right = max(rect[0][0], rect[1][0])
        bottom = max(rect[0][1], rect[1][1])

        for i in range(len(self.initcontour)):
            x = self.initcontour[i][0][0]
            y = self.initcontour[i][0][1]
            _is_in_rect = not(x < left or x > right or y < top or y > bottom)

            if not _is_in_rect:
                continue
            if mode == '-':
                self.selected_tf_table[i] = False
            elif mode == '*':
                self.selected_tf_table[i] &= True
            elif mode == '+':
                self.selected_tf_table[i] = True
            else:
                raise ValueError('unknown mode "{}"'.format(mode))

        self.logger.info('%d selected', self.selected_tf_table.count(True))

    def show_and_interact(self):
        modes = ['*', '+', '-']
        mode = modes[0]
        global ix, iy, drawing, img_tmp
        drawing = False  # true if mouse is pressed
        ix, iy = -1, -1

        def redraw():
            global img_tmp
            img_tmp = self.img.copy()
            self._draw_init_contour(img_tmp)
            self._draw_selected_contour(img_tmp)
            self._draw_mode_indicator(img_tmp, mode)

        def reset():
            # reset to all selected.
            self.selected_tf_table = [True]*len(self.initcontour)
            redraw()

        reset()

        # Mouse callback function
        def on_mouse(event, x, y, flags, param):
            global ix, iy, drawing

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    redraw()
                    cv2.rectangle(img_tmp, (ix, iy), (x, y), (0, 255, 0), 1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                self._update_selected_contour(rect=((ix, iy), (x, y)), mode=mode)
                redraw()
                # cv2.rectangle(img_tmp, (ix, iy), (x, y), (170, 255, 170), 1)

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, on_mouse)

        self.logger.info('Press "r" to reset; "q" to quit; "m" to change select mode')
        while True:
            cv2.imshow(self.name, img_tmp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                # Looping modes
                mode = modes[(modes.index(mode) +1) % len(modes)]
                self.logger.info('selection mode changed to "%s"', mode)
                redraw()
            if k == ord('r'):
                reset()
            if k == ord('q'):
                break

        cv2.destroyWindow(self.name)

    def get_selected(self):
        """Return selected contours"""
        # Applying the T-F table
        l_ = list(filter(lambda i: i[0], zip(self.selected_tf_table, self.initcontour)))
        l_ = [v for e, v in l_]
        # l_ = []
        # if self.selected_tf_table
        # for i,v in enumerate(self.initcontour):
        #     if self.selected_tf_table[i]
        return l_



if __name__ == '__main__':

    test_img = util.resize(cv2.imread('../test_dataset/DSC01332.jpg'), 500)
    c = ContourSelector(test_img, [])
    c.show_and_interact()