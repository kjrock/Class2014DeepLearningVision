import cv2
import numpy as np

img1 = cv2.imread('img/morphological.png')


k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, k)

merged1 = np.hstack((img1, gradient))

cv2.imshow('gradient', merged1)
cv2.waitKey()
cv2.destroyAllWindows()