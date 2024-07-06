import cv2

img = cv2.imread('data/lena.jpg')
cv2.imshow('roi', img)

isDrag = False
x0, y0 = -1, -1
w, h = -1, -1

def roiFunc(event, x, y, flags, param):
    global isDrag, x0, y0

    if event == cv2.EVENT_LBUTTONDOWN:
        isDrag = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrag:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), (0,255,0), 3)
            cv2.imshow('roi', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDrag:
            isDrag = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                roi = img[y0:y0 + h, x0:x0 + w]
                cv2.namedWindow('roiImage', cv2.WINDOW_NORMAL)
                cv2.imshow('roiImage', roi)
                print(roi.shape)
                cv2.resizeWindow('roiImage', roi.shape[1], roi.shape[0])
                cv2.moveWindow('roiImage', 50,50)
                cv2.imwrite('data/roiImage.jpg', roi)
            else:
                cv2.imshow('roi', img)
                print('좌측 상단으로부터 우측 하단으로 드래그를 하세요')

cv2.setMouseCallback('roi', roiFunc)

cv2.waitKey()
cv2.destroyAllWindows()