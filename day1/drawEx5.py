import cv2

img = cv2.imread('img/blank_500.jpg')

cv2.circle(img, (150, 150), 110, (0,0,255))
cv2.circle(img, (400, 150), 70, (0,255,255), -1)

cv2.ellipse(img, (50, 300), (40, 80), 0, 0, 360, (0, 255, 0))
cv2.ellipse(img, (250, 300), (50, 50), 0, 0, 150, (255, 0, 0))
cv2.ellipse(img, (400, 300), (50, 75), 45, 0, 360, (0, 0, 255), 3)

cv2.imshow('circle, ellipse', img)
cv2.waitKey()
cv2.destroyAllWindows()