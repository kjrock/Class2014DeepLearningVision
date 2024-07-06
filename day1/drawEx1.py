import cv2
import numpy as np

# img = np.full((500,500,3), 255, dtype=np.uint8)
img = cv2.imread('img/girl.jpg')

cv2.line(img, (50,50), (450, 50), (255,0,0))
cv2.line(img, (100,100), (400, 100), (0,255,0), 10)
cv2.line(img, (100,200), (400, 200), (0,255,255), 10, cv2.LINE_AA)

cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()