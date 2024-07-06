import cv2
import numpy as np

img1 = cv2.imread('img/morph_dot.png')
img2 = cv2.imread('img/morph_hole.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)

merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))

cv2.imshow('opening closing', merged3)
cv2.waitKey()
cv2.destroyAllWindows()