import cv2
import numpy as np

img = cv2.imread('img/morph_hole.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

dilation = cv2.dilate(img, k)

merged = np.hstack((img, dilation))
cv2.imshow('dilation', merged)
cv2.waitKey()
cv2.destroyAllWindows()