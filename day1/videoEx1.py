import cv2

video_file = 'img/vtest.avi'

cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow(video_file, img)
            if cv2.waitKey(100) != -1:
                break
        else:
            break
else:
    print('VideoCapture를 열수 없습니다.')