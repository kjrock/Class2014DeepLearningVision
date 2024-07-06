import cv2
import numpy as np

lena = cv2.imread('data/lena.jpg')
logo = cv2.imread('data/opencv_logo.png')

print(lena.shape)
print(logo.shape)

logo2 = np.zeros_like(lena)
mask = np.full_like(lena, 255)

height, width = logo.shape[:2]

for i in range(height):
    for j in range(width):
        if logo[i][j][0] == 0 or logo[i][j][1] == 0 or logo[i][j][2] == 0:
            logo2[i][j] = logo[i][j]
            mask[i][j] = [0,0,0]

bitAnd = cv2.bitwise_and(lena, mask)
bitOr = cv2.bitwise_or(bitAnd, logo2)


cv2.imshow('bitAnd', bitAnd)
cv2.imshow('bitOr', bitOr)

# masked = cv2.bitwise_and(lena, mask)
# cv2.imshow('bitwise or img', masked)

cv2.waitKey()
cv2.destroyAllWindows()