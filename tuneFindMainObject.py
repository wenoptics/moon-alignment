import logging

import cv2

from find import FindMainObject

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')

if __name__ == '__main__':
    ori_img = cv2.imread(
        # 'test_dataset/DSC01313.jpg'
        # 'test_dataset/DSC01332.jpg'
        # 'test_dataset/DSC01422.jpg'
        # 'test_dataset/DSC01426.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01364.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01434.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01325.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01367.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01332.jpg'
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01425.jpg'
        'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01344.jpg'
    )

    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('ori', resize(ori_img, 500))
    print('origin img.shape={}'.format(ori_img.shape))
    FindMainObject(force_resize_preview_w=500).run_pipeline_tuning(ori_img)
