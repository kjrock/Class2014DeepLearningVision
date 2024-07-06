import cv2

face_casecade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_casecade.detectMultiScale(gray, scaleFactor=1.3, minSize=(80,80))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi, minSize=(40,40))

            for i, (ex, ey, ew, eh) in enumerate(eyes):
                if i >= 2:
                    break
                cv2.rectangle(img[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)
        cv2.imshow('face detect', img)
    else:
        break

    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()