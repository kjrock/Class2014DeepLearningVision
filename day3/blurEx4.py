import cv2
import numpy as np

img = cv2.imread('img/salt_pepper_noise.jpg')

blur1 = cv2.medianBlur(img, 5)
merged = np.hstack((img, blur1))
cv2.imshow('blur', merged)
cv2.waitKey()
cv2.destroyAllWindows()