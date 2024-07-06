import cv2

img = cv2.imread('img/shapes_donut.png')
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

contour1, hierachy1 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour2, hierachy2 = cv2.findContours(imthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(contour1), len(contour2))

print(hierachy2)

cv2.drawContours(img, contour1, -1, (0, 255, 0), 3)

import numpy as np

for idx, cont in enumerate(contour2):
    color = [int(i) for i in np.random.randint(0,255, 3)]
    cv2.drawContours(img2, contour2, idx, color, 3)
    cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

cv2.imshow('retr_external', img)
cv2.imshow('retr_tree', img2)
cv2.waitKey()
cv2.destroyAllWindows()
