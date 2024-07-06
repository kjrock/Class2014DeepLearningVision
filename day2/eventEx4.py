import cv2

img = cv2.imread('data/blank_500.jpg')
cv2.namedWindow('trackbar')
cv2.imshow('trackbar', img)

def onChange(x):
    #print(x)
    red = cv2.getTrackbarPos('RED', 'trackbar')
    green = cv2.getTrackbarPos('GREEN', 'trackbar')
    blue = cv2.getTrackbarPos('BLUE', 'trackbar')

    img[:] = [blue, green, red]
    cv2.imshow('trackbar', img)


cv2.createTrackbar('RED', 'trackbar', 255, 255, onChange)
cv2.createTrackbar('GREEN', 'trackbar', 255, 255, onChange)
cv2.createTrackbar('BLUE', 'trackbar', 255, 255, onChange)

while True:
    if cv2.waitKey(0) !=-1:
        break

cv2.destroyAllWindows()