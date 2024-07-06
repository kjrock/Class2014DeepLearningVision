import cv2

img = cv2.imread('img/hand.jpg')
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

contour, hierachy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cntr = contour[0]
cv2.drawContours(img, [cntr], -1, (0, 255, 0), 1)

hull = cv2.convexHull(cntr)
cv2.drawContours(img2, [hull], -1, (0, 255, 0), 1)

hull2 = cv2.convexHull(cntr, returnPoints=False)

defects = cv2.convexityDefects(cntr, hull2)
print(defects)

for i in range(defects.shape[0]):
    starp, endp, farthestp, distance = defects[i, 0]
    farthest = tuple(cntr[farthestp][0])
    dist = distance / 256.
    if dist > 1:
        cv2.circle(img2, farthest, 3, (0,0,255), -1)

cv2.imshow('contour', img)
cv2.imshow('convex hull', img2)
cv2.waitKey()
cv2.destroyAllWindows()