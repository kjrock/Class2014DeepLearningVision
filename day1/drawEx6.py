import cv2
import numpy as np

img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255

ptCenter = img.shape[0] // 2, img.shape[1] // 2
size = 200, 100

cv2.ellipse(img, ptCenter, size, 0, 0, 360, (255, 0, 0))
pst1 = cv2.ellipse2Poly(ptCenter, size, 0, 0, 360, delta=45)
print(pst1)

cv2.polylines(img, [pst1], isClosed=True, color=(0,0,255))

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()