import cv2

mask_hannibal = cv2.imread('img/mask_hannibal.png')

def drawImage(src1, src2, x,y):
    rows, cols, channels = src2.shape
    roi = src1[y:y+rows, x:x+cols]

    # 2
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mk', mask)
    cv2.imshow('mk_inv', mask_inv)

    # 3
    src1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    cv2.imshow('src1_bg', src1_bg)

    # 4
    src2_fg = cv2.bitwise_and(src2, src2, mask=mask)
    cv2.imshow('src2_fg', src2_fg)

    # 5
    dst = cv2.bitwise_or(src1_bg, src2_fg)
    # cv2.imshow('dst', dst)

    # 6
    src1[y:y+rows, x:x+cols] = dst
    cv2.imshow('result', src1)

face_casecade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_casecade.detectMultiScale(gray, scaleFactor=1.3, minSize=(80,80))

        for (x, y, w, h) in faces:
            x = x + (w//10)
            w = w - (w//5)
            y = y + (h//3)
            h = h - (h//3)
            mask = cv2.resize(mask_hannibal, (w, h))
            drawImage(img, mask, x, y)
    else:
        break

    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()
