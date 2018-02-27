import logging

import cv2
import os

from find import draw_moon, find_moon, FindMainObject, FindCircle
from utils import util

PREVIEW_WIDTH = 500

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(name)-12s %(levelname)-8s %(message)s')

# images = [
#     'test_dataset/DSC01313.jpg',
#     'test_dataset/DSC01332.jpg',
#     'test_dataset/DSC01422.jpg',
#     'test_dataset/DSC01426.jpg',
# ]
# images = open('test_dataset/test_datasets_all.txt', encoding='utf-8').read().split()
images = open('test_dataset/test_datasets_D810.txt', encoding='utf-8').read().split()

padding_top = 500
padding_bottom = 500
padding_left = 700
padding_right = 700
save_cropped_folder = 'test_dataset/aligned/'


def crop_moon(img, moon):
    # crop img
    crop_img = img[moon.y - padding_top:moon.y + padding_bottom, moon.x - padding_left:moon.x + padding_right]
    # todo logger.error('[E] crop image "{}" failed'.format(img_path))
    return crop_img


f0 = FindMainObject()
f1 = FindCircle()


def interactive(img_ori, path):
    def _manually_tune():
        f0.force_resize_preview_w = 300
        f1.force_resize_preview_w = 200
        return find_moon(img_ori, f0, f1, tune=True)

    def _process_moon(moon, path):
        if not moon:
            print('moon not detected in {}, press "d" to manually tune.'.format(path))
            cv2.imshow('ori - {}'.format(path), util.resize(img_ori, PREVIEW_WIDTH))
            retkey = cv2.waitKey()
            if 'd' == chr(retkey & 255):
                cv2.destroyAllWindows()
                newmoon = _manually_tune()
                _process_moon(newmoon, path)
            return

        img_draw = img_ori.copy()
        draw_moon(img_draw, moon)

        img_crop = crop_moon(img_ori, moon)
        cv2.imshow('crop preview - {}'.format(path), util.resize(crop_moon(img_draw, moon), PREVIEW_WIDTH))

        while True:
            logger.info('press "y" or "n" to save or skip crop image. "d" to manually tune.')
            retkey = cv2.waitKey()
            if 'y' == chr(retkey & 255):
                saved_path = os.path.join(save_cropped_folder, os.path.basename(path))
                cv2.imwrite(saved_path, img_crop)
                logger.info('cropped image saved to {}'.format(saved_path))
                break

            elif 'n' == chr(retkey & 255):
                logger.info('image "{}" will not crop'.format(path))
                break

            elif 'd' == chr(retkey & 255):
                logger.info('manually tune for image "{}" started'.format(path))
                cv2.destroyAllWindows()
                newmoon = _manually_tune()
                _process_moon(newmoon, path)
                break
        cv2.destroyAllWindows()

    moon = find_moon(img_ori, f0, f1)
    _process_moon(moon, path)


for img_path in images:
    try:
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            print('load image "%s" failed' % img_path)
            continue
        # img = cv2.resize(ori_img, (0, 0), fx=1, fy=1)
        img = ori_img.copy()
        interactive(img, img_path)

    except:
        logger.error('error occur when processing image "{}"'.format(img_path), exc_info=True)
