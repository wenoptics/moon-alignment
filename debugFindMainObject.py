from time import time

import cv2
import numpy as np

from find import find_main_object
from utils.util import myTuneWindow


def loadimg():
    ori_img = cv2.imread(
        'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
    print('origin img.shape={}'.format(ori_img.shape))

    myTuneWindow(find_main_object, ori_img,
                 valMedianBlur=(1, 30, 2),
                 valKernelOpen=(1, 100, 2),
                 valKernelClose=(1, 100, 2),
                 valAdaptiveThreshold1=(1, 255),
                 valAdaptiveThreshold2=(1, 255),
                 )


if __name__ == '__main__':
    loadimg()
