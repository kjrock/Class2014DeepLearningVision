import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('camera', img)
            if cv2.waitKey(400) != -1:
                break
        else:
            print('no frame')
            break
else:
    print('카메라 장치가 없습니다.')

cap.release()
cv2.destroyAllWindows()