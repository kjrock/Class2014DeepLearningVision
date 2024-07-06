import numpy as np
import cv2

img = cv2.imread('data/fish.jpg')
rows, cols = img.shape[:2]

m_small = np.float32([[0.5, 0, 0],
                      [0, 0.5, 0]])

m_big = np.float32([[2, 0, 0],
                      [0, 2, 0]])

dst1 = cv2.warpAffine(img, m_small, (int(cols * 0.5), int(rows * 0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(cols * 2), int(rows * 2)))

cv2.imshow('original', img)
cv2.imshow('small', dst1)
cv2.imshow('big', dst2)

cv2.waitKey()
cv2.destroyAllWindows()