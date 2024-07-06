import cv2
import numpy as np

img = cv2.imread('img/morph_dot.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

eroision = cv2.erode(img, k)

merged = np.hstack((img, eroision))
cv2.imshow('eroide', merged)
cv2.waitKey()
cv2.destroyAllWindows()