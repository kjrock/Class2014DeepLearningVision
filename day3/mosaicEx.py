import cv2

img = cv2.imread('img/cat.jpg')

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

mos = mosaic(img, (0,50,350,350), 10)
cv2.imshow('img', img)
cv2.imshow('mosaic', mos)
cv2.waitKey()
cv2.destroyAllWindows()