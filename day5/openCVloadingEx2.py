import cv2
import numpy as np

net = cv2.dnn.readNet('dnn/MNIST_CNN.pb')

model = cv2.dnn_ClassificationModel(net)
model.setInputParams(scale=1/255, size=(28,28))

colors ={'black':(0,0,0), 'white':(255,255,255), 'blue':(255,0,0), 'red':(0,0,255)}

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(src, (x,y), 10, colors['black'], -1)
            cv2.imshow('image', src)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(src, (x,y), 10, colors['white'], -1)
            cv2.imshow('image', src)


def makeSquareImage(img):
    height, width = img.shape[:2]
    if width > height:
        resImg = np.zeros(shape=(width, width), dtype=np.uint8)
        y0 = (width - height) // 2
        resImg[y0:y0+height, :] = img
    elif width < height:
        x0 = (height - width) // 2
        resImg = np.zeros(shape=(height, height), dtype=np.uint8)
        resImg[:, x0:x0+width] = img
    else:
        resImg = img
    return resImg


src = np.full((600,600,3), colors['white'], dtype=np.uint8)
cv2.imshow('image', src)
cv2.setMouseCallback('image', onMouse)

kernel = np.ones((7,7), np.uint8)
x_img = np.zeros(shape=(28,28), dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    key = cv2.waitKey(25)
    if key == 27:
        break
    elif key == 32: #space bar
        src[:,:] = colors['white']
        cv2.imshow('image', src)

    elif key == 13: #enter
        print('분류!!!!!!!')
        dst = src.copy()
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, th_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        th_img = cv2.dilate(th_img, kernel)
        contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            x, y, width, height = cv2.boundingRect(cnt)
            area = width * height
            if area < 1000: continue

            cv2.rectangle(dst, (x, y), (x+width, y+height), colors['red'], 2)
            img = th_img[y:y+height, x:x+width]
            img = makeSquareImage(img)

            img = cv2.resize(img, dsize=(20, 20), interpolation=cv2.INTER_AREA)
            x_img[:,:] = 0
            x_img[4:24, 4:24] = img

            blob = cv2.dnn.blobFromImage(x_img, scalefactor=1/255)

            net.setInput(blob)
            y_out = net.forward()
            digit = np.argmax(y_out, axis=1)[0]
            print('net forward() digit:',digit)

            digit, prob = model.classify(x_img)
            print(f'model.classify() digit:{digit}, prob:{prob:.2f}')

            cv2.putText(dst, str(digit), (x, y), font, 2, colors['blue'], 3)


        cv2.imshow('image', dst)

cv2.destroyAllWindows()