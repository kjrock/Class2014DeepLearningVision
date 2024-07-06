import cv2
import numpy as np

def draw_func(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            cv2.rectangle(img, (x, y), (x+20, y+20), (0,0,255), 2)
        else:
            cv2.rectangle(img, (x, y), (x+20, y+20), (0,255,0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(img, (x, y), 10, (255, 0, 0), 2)
        else:
            cv2.circle(img, (x, y), 20, (255, 0, 0), -1)

img = np.full((512,512,3), 255, dtype=np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_func)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()
