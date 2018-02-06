import logging

import cv2

from find import FindMainObject, FindCircle

logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')


if __name__ == '__main__':

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
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01353.jpg',
        # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01364.jpg',
        'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01367.jpg',
    )
    # ori_img = cv2.resize(ori_img, (0, 0), fx=0.1, fy=0.1)
    print('origin img.shape={}'.format(ori_img.shape))
    roi, x, y = FindMainObject().run_pipeline_final(ori_img)
    FindCircle(force_resize_preview_w=400).run_pipeline_tuning(roi)
