import cv2

def mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect

    w = x2 - x1
    h = y2 - y1

    i_rect = img[y1:y2, x1:x2]
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2

face_casecade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

img = cv2.imread('img/family.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_casecade.detectMultiScale(gray, minSize=(150, 150))

for (x, y, w, h) in faces:
    img = mosaic(img, (x,y,x+w,y+h), 10)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()