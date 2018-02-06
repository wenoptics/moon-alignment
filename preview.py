import cv2

from find import find_moon, draw_moon, FindMainObject, FindCircle
from utils import util

PREVIEW_WIDTH = 500

test_img = [
    # 'test_dataset/DSC01313.jpg',
    # 'test_dataset/DSC01332.jpg',
    # 'test_dataset/DSC01422.jpg',
    # 'test_dataset/DSC01426.jpg',
    # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01313.jpg',
    # 'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01321.jpg',
    'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01325.jpg',
    'E:\\tmp\\Eclipse-20180131\\chosen\\moon\\output\\JPEG\\DSC01332.jpg',
]
# test_img = open('test_dataset/test_datasets.txt', encoding='utf-8').read().split()

f0 = FindMainObject()
f1 = FindCircle()

for img_path in test_img:
    ori_img = cv2.imread(img_path)
    # img = cv2.resize(ori_img, (0, 0), fx=0.3, fy=0.3)
    img = ori_img.copy()

    moon = find_moon(img, f0, f1)
    if not moon:
        print('moon not detected in {}'.format(img_path))
        cv2.imshow('ori - {}'.format(img_path), util.resize(img, PREVIEW_WIDTH))
        continue

    draw_moon(img, moon)

    cv2.imshow('moondetect - {}'.format(img_path), util.resize(img, PREVIEW_WIDTH))
    # cv2.imwrite('img/moondetect_preview.jpg'.format(img_path), img)
    cv2.waitKey()
    cv2.destroyAllWindows()

