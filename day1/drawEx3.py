import cv2
import numpy as np

img = cv2.imread('img/blank_500.jpg')

pts1 = np.array([[59,59],[150,150],[110,140],[210,250]], dtype=np.int32)
pts2 = np.array([[350,50],[250,200],[450,200]], dtype=np.int32)

cv2.polylines(img, [pts1], False, (255,0,0))
cv2.polylines(img, [pts2], True, (0,0,255), 10)

cv2.imshow('polyline', img)
cv2.waitKey()
cv2.destroyAllWindows()