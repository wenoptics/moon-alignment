from time import time

import cv2
import numpy as np

from find import find_circle, find_main_object
from utils.util import myTuneWindow


def loadimg():
    ori_img = cv2.imread(
        # 'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'

        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01313.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01321.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01325.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01332.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01350.jpg',
        'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01353.jpg',
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.1, fy=0.1)
    print('origin img.shape={}'.format(ori_img.shape))
    roi, x, y = find_main_object(ori_img, show_debug_preview=False)
    myTuneWindow(find_circle, roi,
                 # valMedianBlur=(1, 30, 2),
                 valKernelOpen=(1, 100, 2),
                 valKernelClose=(1, 100, 2),
                 valHoughParam1=(1, 300),
                 valHoughParam2=(1, 300),
                 valHoughMinDist=(1, 1000),
                 # valAdaptiveThreshold=(0,255),
                 valBlfColor=(0, 1250),
                 valBlfSpace=(0, 1250),
                 valBlfD=(0, 30),
                 )


if __name__ == '__main__':
    loadimg()
