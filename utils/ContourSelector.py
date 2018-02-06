import cv2

from utils import util


class ContourSelector:
    def __init__(self, img, initcontour, name='contour selector'):
        self.name = name
        self.initcontour = initcontour
        self.selectedpoints = []
        self.img = img.copy()

    def show(self):

        global ix, iy, drawing, img_tmp
        drawing = False  # true if mouse is pressed
        ix, iy = -1, -1

        img_tmp = self.img.copy()

        def reset():
            global img_tmp
            img_tmp = self.img.copy()

        # mouse callback function
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
                reset()
                cv2.rectangle(img_tmp, (ix, iy), (x, y), (170, 255, 170), 1)

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, on_mouse)

        print('Press "q" to quit')
        while True:
            cv2.imshow(self.name, img_tmp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                pass
            if k == ord('q'):
                break

        cv2.destroyWindow(self.name)


if __name__ == '__main__':

    test_img = util.resize(cv2.imread('../test_dataset/DSC01332.jpg'), 500)
    c = ContourSelector(test_img, None)
    c.show()