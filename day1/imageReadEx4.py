import cv2

img = cv2.imread('img/girl.jpg')
im2 = cv2.resize(img,  (600, 300))
cv2.imwrite('img/out-resize.png', im2)

cv2.imshow('IMG', img)
cv2.imshow('IMG2', im2)
cv2.waitKey()
cv2.destroyAllWindows()
