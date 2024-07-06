import cv2
import numpy as np

img1 = cv2.imread('data/wing_wall.jpg')
img2 = cv2.imread('data/yate.jpg')

alpha = 0.5

blended = img1 * alpha + img2 * (1 - alpha)
blended = blended.astype(np.uint8)
cv2.imshow('blended', blended)

dst = cv2.addWeighted(img1, alpha, img2, (1 - alpha), 0)
cv2.imshow('addWeighted', dst)

cv2.waitKey()
cv2.destroyAllWindows()