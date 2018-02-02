import logging

import cv2
import os

from find import draw_moon, find_moon, resize

PREVIEW_WIDTH = 500

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')

images = [
    'test_dataset/DSC01313.jpg',
    'test_dataset/DSC01332.jpg',
    'test_dataset/DSC01422.jpg',
    'test_dataset/DSC01426.jpg',
]
images = open('test_dataset/test_datasets_all.txt', encoding='utf-8').read().split()

padding_top = 500
padding_bottom = 500
padding_left = 700
padding_right = 700
save_cropped_folder = './'

for img_path in images:
    try:
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            print('load image "%s" failed' % img_path)
            continue
        # img = cv2.resize(ori_img, (0, 0), fx=1, fy=1)
        img = ori_img.copy()

        moon = find_moon(img)
        if not moon:
            print('moon not detected in {}, press any key to continue'.format(img_path))
            cv2.imshow('ori - {}'.format(img_path), resize(img, PREVIEW_WIDTH))
            cv2.waitKey()
            cv2.destroyAllWindows()
            continue

        draw_img = img.copy()
        draw_moon(draw_img, moon)

        # crop img
        crop_img_preview = draw_img[moon.y-padding_top:moon.y+padding_bottom, moon.x-padding_left:moon.x+padding_right]
        # todo logger.error('[E] crop image "{}" failed'.format(img_path))
        cv2.imshow('crop preview - {}'.format(img_path), resize(crop_img_preview, PREVIEW_WIDTH))

        crop_img = img[moon.y-padding_top:moon.y+padding_bottom, moon.x-padding_left:moon.x+padding_right]

        logger.info('press "y" or "n" to save or skip crop image.')

        retkey = cv2.waitKey()
        if 'y' == chr(retkey & 255):
            saved_path = os.path.join(save_cropped_folder, os.path.basename(img_path))
            cv2.imwrite(saved_path, crop_img)
            logger.info('cropped image saved to {}'.format(saved_path))
        elif 'n' == chr(retkey & 255):
            logger.info('image "{}" will not crop'.format(img_path))
        else:
            logger.info('press "y" or "n"')

        cv2.destroyAllWindows()

    except:
        logger.error('error occur when processing image "{}"'.format(img_path), exc_info=True)



