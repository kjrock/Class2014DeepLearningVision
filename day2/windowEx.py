import cv2

file_path = 'data/girl.jpg'
img = cv2.imread(file_path)
img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('original')
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.imshow('original', img)
cv2.imshow('gray', img_gray)

cv2.resizeWindow('original', 400, 400)
cv2.resizeWindow('gray', 500, 500)

cv2.moveWindow('original', 0, 0)
cv2.moveWindow('gray', 100, 100)

bValue = False
while True:
    if cv2.waitKey(0) == 27: #esc
        cv2.destroyWindow('gray')
        if bValue:
            break
        bValue = True

    if cv2.waitKey(0) == 13: #enter
        cv2.destroyWindow('original')
        if bValue:
            break
        bValue = True