import cv2

video_file = 'img/vtest.avi'

cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('width', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    delay = int(1000/fps)
    print(f'fps:{fps}, delay:{delay}ms')

    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow(video_file, img)
            if cv2.waitKey(100) != -1:
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()