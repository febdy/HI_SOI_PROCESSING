import numpy as np  # matrix operations (ie. difference between two matricies)
import cv2  # (OpenCV) computer vision functions (ie. tracking)

OUTLINE = True
LRG_ONLY = True

# window to hold the trackbar
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# create trackbar
cv2.createTrackbar('Thresh', 'image', 0, 255, lambda x: None)

image = cv2.imread('C:/Users/BIT-USER/Desktop/python_workplace/face.jpg')

while True:
    thresh_min = cv2.getTrackbarPos('Thresh', 'image')

    contour_img = image.copy()
    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
    ret, contour_img_thresh = cv2.threshold(contour_img, thresh_min, 255, 0)
    im2, contours, hierarchy = cv2.findContours(contour_img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if LRG_ONLY:
        cnts = [x for x in contours if cv2.contourArea(x) > 20000]
    else:
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    if OUTLINE:
        # Draw only outlines
        contour_img_display = cv2.drawContours(image.copy(), cnts, -1, (238, 255, 0), 2)
    else:
        # Draw filled contours
        contour_img_display = cv2.drawContours(image.copy(), cnts, -1, (238, 255, 0), -1)

    contour_img_display = cv2.cvtColor(contour_img_display, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', contour_img_display)
    cv2.imshow('thresh', contour_img_thresh)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break  # ESC pressed

cv2.destroyAllWindows()
