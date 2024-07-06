# 1.흰색 배경이 있는 윈도우에 아래의 조건에 맞게 이벤트 처리를 하시오.
# ㄱ. 마우스 좌클릭시 초록색으로 채워진 사각형을 이벤트 위치에 출력
# ㄴ. 마우스 좌클릭과 ctrl키를 같이 눌렀을경우에는 빨간색 사각형을 이벤트 위치에 출력
# ㄷ. 마우스 우클릭시 파란색으로 채워진 원을 이벤트 위치에 출력
# ㄹ. 마우스 우클릭과 shift키를 같이 눌렀을때 파란색의 원을 이벤트 위치에 출력
# ㅁ. esc 키를 누르면 해당 프로그램 종료

import cv2
import numpy as np

img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
cv2.imshow('quiz', img)

def callMouseProcess(event, x, y, flags, param):
    ccolor = (0, 0, 0)
    #print(event, x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            ccolor = (0, 0, 255)
        else:
            ccolor = (0, 255, 0)

        cv2.rectangle(img, (x - 15, y - 15), (x + 15, y + 15), ccolor)

    elif event == cv2.EVENT_RBUTTONDOWN:
        ccolor = (255, 0, 0)
        thick = 0
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
           thick = -1

        cv2.circle(img, (x, y), 20, ccolor, thick)
    cv2.imshow('quiz', img)

cv2.setMouseCallback('quiz', callMouseProcess)

while True:
    if cv2.waitKey(0) != -1:
        break

cv2.destroyAllWindows()