import cv2

from findcircle import find_circle

test_img = [
    'test_dataset/DSC01313.jpg',
    'test_dataset/DSC01332.jpg',
    'test_dataset/DSC01422.jpg',
    'test_dataset/DSC01426.jpg',
]
test_img = open('test_dataset/test_datasets.txt', encoding='utf-8').read().split()

for img_path in test_img:
    ori_img = cv2.imread(img_path)
    img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)

    circles = find_circle(img, show_debug_preview=False)
    if not circles:
        print('moon not detected in {}'.format(img_path))
        cv2.imshow('ori - {}'.format(img_path), img)
        continue

    for i in circles:
        # draw the outer circle
        cv2.circle(img, (i.x, i.y), i.r, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i.x, i.y), 5, (0, 0, 255), 1)

    cv2.imshow('moondetect - {}'.format(img_path), img)
    # cv2.imwrite('img/moondetect_preview.jpg'.format(img_path), img)

cv2.waitKey()
cv2.destroyAllWindows()
