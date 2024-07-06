import cv2

cap = cv2.VideoCapture(0)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out1 = cv2.VideoWriter('img/record1.avi', fourcc, 20.0, frame_size)
out2 = cv2.VideoWriter('img/record1.avi', fourcc, 20.0, frame_size, isColor=False)


while True:
    ret, frame = cap.read()
    if ret:
        out1.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        out2.write(gray)
        cv2.imshow('color', frame)
        cv2.imshow('gray', gray)

        if cv2.waitKey(25) != -1:
            break
    else:
        print('no frame')
        break


cap.release()
cv2.destroyAllWindows()