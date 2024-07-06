import cv2

img = cv2.imread('img/blank_500.jpg')
cv2.imshow('event2', img)

def callMouseProcess(event, x, y, flags, param):
    ccolor = (0, 0, 0)
    #print(event, x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            ccolor = (0, 0, 255)
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            ccolor = (0, 255, 0)
        cv2.circle(img, (x, y), 20, ccolor, -1)
        cv2.imshow('event2', img)

cv2.setMouseCallback('event2', callMouseProcess)

while True:
    if cv2.waitKey(0) != -1:
        break

cv2.destroyAllWindows()