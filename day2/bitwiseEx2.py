import cv2
import numpy as np

img = cv2.imread('data/lena.jpg')
mask = np.zeros_like(img)
cv2.circle(mask, (280, 280), 100, (255, 255, 255), -1)

masked = cv2.bitwise_and(img, mask)

cv2.imshow('mask', mask)
cv2.imshow('img', img)
cv2.imshow('bitwise and img', masked)

cv2.waitKey()
cv2.destroyAllWindows()