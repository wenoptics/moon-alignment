import logging

import cv2
import os

from findcircle import find_circle

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
        # img = cv2.resize(ori_img, (0, 0), fx=1, fy=1)
        img = ori_img.copy()

        circles = find_circle(img, show_debug_preview=False)
        if not circles:
            logger.warning('moon not detected in {}'.format(img_path))
            cv2.imshow('ori - {}'.format(img_path), img)
            continue
        if len(circles) > 1:
            logger.warning('[W] multiple circles detect')

        draw_img = img.copy()
        for i in circles:
            # draw the outer circle
            cv2.circle(draw_img, (i.x, i.y), i.r, (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(draw_img, (i.x, i.y), 5, (0, 0, 255), 1)

            # crop img
            crop_img_preview = draw_img[i.y-padding_top:i.y+padding_bottom, i.x-padding_left:i.x+padding_right]
            # todo logger.error('[E] crop image "{}" failed'.format(img_path))
            cv2.imshow('crop preview - {}'.format(img_path), crop_img_preview)

            crop_img = img[i.y-padding_top:i.y+padding_bottom, i.x-padding_left:i.x+padding_right]

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

            break  # only process the first one
    except:
        logger.error('error occur when processing image "{}"'.format(img_path), exc_info=True)



