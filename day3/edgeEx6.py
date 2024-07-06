import cv2
import numpy as np

img = cv2.imread('img/sudoku.jpg')

edge = cv2.Canny(img, 100, 200)

cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey()
cv2.destroyAllWindows()