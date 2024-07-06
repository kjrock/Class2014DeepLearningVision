import numpy as np
import cv2

img = cv2.imread('data/fish.jpg')
rows, cols = img.shape[:2]

pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

cv2.circle(img, (100, 50), 5, (255, 0, 0), -1)
cv2.circle(img, (200, 50), 5, (255, 255, 0), -1)
cv2.circle(img, (100, 200), 5, (255, 0, 255), -1)

mtrx = cv2.getAffineTransform(pts1, pts2)

dst1 = cv2.warpAffine(img, mtrx, (int(cols * 2), rows))

cv2.imshow('original', img)
cv2.imshow('affine', dst1)

cv2.waitKey()
cv2.destroyAllWindows()