import cv2

img = cv2.imread('img/girl.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('IMG', img)
cv2.waitKey()
cv2.destroyAllWindows()
