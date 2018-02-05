import logging

import cv2

from find import FindMainObject

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')

img_ori = cv2.imread('E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01425.jpg')
FindMainObject(force_resize_preview_w=500).run_pipeline_tuning(img_ori)