from time import time

import cv2
import numpy as np

from findcircle import find_circle
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

    myTuneWindow(find_circle, ori_img,
                 valMedianBlur=(1, 30, 2),
                 valKernelOpen=(1, 100, 2),
                 valKernelClose=(1, 100, 2),
                 valAdaptivateThreshold=(1, 255),
                 valHoughParam1=(1, 300),
                 valHoughParam2=(1, 300),
                 valHoughMinDist=(1, 1000),
                 )


if __name__ == '__main__':
    loadimg()
