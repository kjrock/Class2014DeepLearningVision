import cv2

img_file = 'img/lena.jpg'
img = cv2.imread(img_file)
x, y = 100, 100
cv2.imshow('img', img)

while True:
    cv2.moveWindow('img', x, y)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('j'):
        x -= 10
    elif key == ord('l'):
        x += 10
    elif key == ord('i'):
        y -= 10
    elif key == ord('m'):
        y += 10
    elif key == ord('q'):
        break
cv2.destroyAllWindows()