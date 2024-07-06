import cv2

face_casecade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

img = cv2.imread('img/girl.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_casecade.detectMultiScale(gray, minSize=(150, 150))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, minSize=[120,120])

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()