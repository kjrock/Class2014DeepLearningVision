import cv2
import numpy as np

img = cv2.imread('img/sudoku.jpg')

gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1],[1]])

edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

merged = np.hstack((img, edge_gx, edge_gy))
cv2.imshow('blur', merged)
cv2.waitKey()
cv2.destroyAllWindows()