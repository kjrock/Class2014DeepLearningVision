import numpy as np
import cv2

img = cv2.imread('data/fish.jpg')
rows, cols = img.shape[:2]

dx, dy = 100, 40

mtrx = np.float32([[1, 0, dx],
                   [0, 1, dy]])

dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))

cv2.imshow('original', img)
cv2.imshow('trans', dst)
cv2.imshow('trans2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()