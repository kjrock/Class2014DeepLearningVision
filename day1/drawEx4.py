import cv2

img = cv2.imread('img/blank_500.jpg')

cv2.putText(img, 'plain text', (50,20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0))
cv2.putText(img, 'simplex text', (50,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
cv2.putText(img, 'duplex text', (50,140), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255))
cv2.putText(img, 'plain text', (50,250), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 2, (0,0,255))

cv2.imshow('put text', img)
cv2.waitKey()
cv2.destroyAllWindows()