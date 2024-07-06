import cv2
import numpy as np

img = cv2.imread('img/lena.jpg')

kernel = np.ones((5,5)) / 5 ** 2
print(kernel)

blured = cv2.filter2D(img, -1, kernel)

cv2.imshow('original', img)
cv2.imshow('avg blur', blured)
cv2.waitKey()
cv2.destroyAllWindows()