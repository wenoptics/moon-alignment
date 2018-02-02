import logging

import cv2

from find import find_circle, find_main_object
from utils.util import myTuneWindow

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')


def tune_find_circle(img):
    return myTuneWindow(find_circle, img,
                        # valMedianBlur=(1, 30, 2),
                        valKernelOpen=(1, 100, 2),
                        valKernelClose=(1, 100, 2),
                        valHoughParam1=(1, 300),
                        valHoughParam2=(1, 300),
                        valHoughMinDist=(1, 1000),
                        valAdaptiveThreshold=(0,255),
                        valBlfColor=(0, 1250),
                        valBlfSpace=(0, 1250),
                        valBlfD=(0, 30),
                        )


if __name__ == '__main__':

    ori_img = cv2.imread(
        # 'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'

        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01313.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01321.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01325.jpg',
        'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01332.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01350.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01353.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01364.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01367.jpg',
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.1, fy=0.1)
    print('origin img.shape={}'.format(ori_img.shape))
    roi, x, y = find_main_object(ori_img, show_debug_preview=False)
    tune_find_circle(roi)
