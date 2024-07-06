import cv2
import numpy as np

img = cv2.imread('img/gaussian_noise.jpg')

k1 = np.array([[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]]) * (1 / 16)

blur1 = cv2.filter2D(img, -1, k1)
k2 = cv2.getGaussianKernel(3, 0)
print(k2)

f = k2 * k2.T
print(f)

blur2 = cv2.filter2D(img, -1, f)

blur3 = cv2.GaussianBlur(img, (3,3), 0)

merged = np.hstack((img, blur1, blur2, blur3))
cv2.imshow('blur', merged)
cv2.waitKey()
cv2.destroyAllWindows()