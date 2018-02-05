import logging
from timeit import timeit

import cv2

from find import FindMainObject


img_ori = cv2.imread('E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01425.jpg')
t = FindMainObject(force_resize_preview_w=500)

# t.run_pipeline_final(img_ori)
# for _ in range(5):
#     print(timeit('t.run_pipeline_final(img_ori)', number=1, globals=globals()))

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')
t.run_pipeline_tuning(img_ori)

for _ in range(5):
    print(timeit('t.run_pipeline_final(img_ori)', number=1, globals=globals()))