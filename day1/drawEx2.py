import cv2

img = cv2.imread('img/blank_500.jpg')

cv2.rectangle(img, (50,50), (150,250), (255,0,0))
cv2.rectangle(img, (300,400), (100,100), (0,0,255), 10,cv2.LINE_4)
cv2.rectangle(img, (300,400), (200,200), (0,255,0), -1)

cv2.imshow('rectangle', img)
cv2.waitKey()
cv2.destroyAllWindows()