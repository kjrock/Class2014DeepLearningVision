import cv2
import numpy as np

img = cv2.imread('img/lena.jpg')

blur1 = cv2.blur(img, (10, 10))
blur2 = cv2.boxFilter(img, -1, (10, 10))

merged = np.hstack((img, blur1, blur2))
cv2.imshow('blur', merged)
cv2.waitKey()
cv2.destroyAllWindows()