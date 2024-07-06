import numpy as np
import cv2

img = cv2.imread('data/fish.jpg')
rows, cols = img.shape[:2]

d45 = 45.0 * np.pi / 180
d90 = 90.0 * np.pi / 180

m45 = np.float32([[np.cos(d45), -1 * np.sin(d45), rows//2],
                  [np.sin(d45), np.cos(d45), -1 * cols //4]])

m90 = np.float32([[np.cos(d90), -1 * np.sin(d90), rows],
                  [np.sin(d90), np.cos(d90), 0]])

dst1 = cv2.warpAffine(img, m45, (cols, rows))
dst2 = cv2.warpAffine(img, m90, (cols, rows))

cv2.imshow('original', img)
cv2.imshow('m45', dst1)
cv2.imshow('m90', dst2)

cv2.waitKey()
cv2.destroyAllWindows()