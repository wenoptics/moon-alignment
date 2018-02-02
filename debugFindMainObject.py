import logging

import cv2

from find import find_main_object, resize
from utils.util import myTuneWindow

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')


def loadimg():
    ori_img = cv2.imread(
        # 'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01364.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01434.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01325.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01367.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01332.jpg',
        'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01425.jpg',
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow('ori', resize(ori_img, 500))
    print('origin img.shape={}'.format(ori_img.shape))

    myTuneWindow(find_main_object, ori_img,
                 valMedianBlur=(1, 30, 2),
                 valKernelOpen=(1, 100, 2),
                 valKernelClose=(1, 100, 2),
                 valAdaptiveThreshold1=(1, 255),
                 # valAdaptiveThreshold2=(1, 255),
                 # valAdpBSize=(1, 30, 2),
                 # valAdpC=(1, 30),
                 valUpperL=(0,255),
                 valLowerL=(0,255),
                 valThr=(0,255),
                 )


if __name__ == '__main__':
    loadimg()
