import cv2

from tuneFindCircle import tune_find_circle
from tuneFindMainObject import turn_find_main_object


def tune_find_moon(img_path):
    roi, rx, ry = turn_find_main_object(img_path)
    cv2.destroyAllWindows()
    if roi is None:
        print('Failed to locate a main object, aborted.')
        return None
    circles = tune_find_circle(roi)
    cv2.destroyAllWindows()

    if not circles:
        return None

    ret_circle = circles[0]
    if len(circles) > 1:
        print('Warning: detected more than one circle, use the largest one')

        largest_r = 0
        for c in circles:
            if c.r > largest_r:
                ret_circle = c
                largest_r = c.r

    # Offset the roi
    ret_circle.x += rx
    ret_circle.y += ry

    return ret_circle


if __name__ == '__main__':
    tune_find_moon(
        # "E:/tmp/Eclipse-20180131/D810/jpg/DSC_3363.jpg"
        "E:/tmp/Eclipse-20180131/D810/jpg/DSC_3357.jpg"
    )