import cv2
import numpy as np

img = cv2.imread('img/lena.jpg')
rows, cols = img.shape[:2]

mapy, mapx = np.indices((rows, cols), dtype=np.float32)


mapx = 2 * mapx / (cols - 1) - 1
mapy = 2 * mapy / (rows - 1) - 1

print(mapy)
print()
print(mapx)
print()

r, theta = cv2.cartToPolar(mapx, mapy)
print(r)
print()
print(theta)

exp = 0.5
scale = 1

r[r < scale] = r[r < scale] ** exp

mapx, mapy = cv2.polarToCart(r, theta)

mapx = ((mapx + 1) * cols - 1) / 2
mapy = ((mapy + 1) * rows - 1) / 2

distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow('original', img)
cv2.imshow('distorted', distorted)
cv2.waitKey()
cv2.destroyAllWindows()
