import numpy as np  # matrix operations (ie. difference between two matricies)
import cv2  # (OpenCV) computer vision functions (ie. tracking)


bg_img = cv2.imread('C:/Users/BIT-USER/Desktop/python_workplace/interviewers.jpg')
current_frame_img = cv2.imread('C:/Users/BIT-USER/Desktop/python_workplace/interviewers_2.jpg')

diff = cv2.absdiff(bg_img, current_frame_img)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

mask_indexes = mask_thresh > 0

foreground = np.zeros_like(current_frame_img, dtype=np.uint8)

for i, row in enumerate(mask_indexes):
    foreground[i, row] = current_frame_img[i, row]

cv2.imshow('bg_img', bg_img)
cv2.imshow('current_frame_img', current_frame_img)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)
cv2.imshow('mask_threshold', mask_thresh)
cv2.imshow('foreground', foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()
