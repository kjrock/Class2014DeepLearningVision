import cv2

img = cv2.imread('img/bad_rect.png')
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

contour, hierachy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contour)

contour = contour[0]
epsilon = 0.05 * cv2.arcLength(contour, True)
print(epsilon)
approx = cv2.approxPolyDP(contour, epsilon, True)

cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
cv2.drawContours(img2, [approx], -1, (0, 255, 0), 3)

cv2.imshow('contour', img)
cv2.imshow('approx', img2)
cv2.waitKey()
cv2.destroyAllWindows()