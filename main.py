import cv2
import numpy as np

ori_img = cv2.imread('test_dataset/DSC01332.jpg')
ori_img = cv2.resize(ori_img, (0, 0), fx=0.2, fy=0.2)
gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120,
                           param1=100, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(ori_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(ori_img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite("planets_circles.jpg", ori_img)
cv2.imshow("HoughCirlces", ori_img)
cv2.waitKey()
cv2.destroyAllWindows()
