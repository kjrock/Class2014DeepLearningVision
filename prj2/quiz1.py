import cv2
import numpy as np

img1 = cv2.imread('img/lion_face.jpg')
img2 = cv2.imread('img/man_face.jpg')

def drawImage(alpha=0.0):
    blended = img1 * alpha + img2 * (1 - alpha)
    blended = blended.astype(np.uint8)
    cv2.imshow('blended', blended)

def onChange(pos):
    print(pos)
    drawImage(pos/100)

drawImage()
cv2.createTrackbar("alpha", "blended", 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()