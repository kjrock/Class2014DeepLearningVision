import cv2
import numpy as np

img = cv2.imread('data/paper.jpg')
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

def perspectiveFunc(event, x, y, flags, param):
    global pts_cnt

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow('perspectiveEx', draw)

        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        if pts_cnt == 4:

            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            topLeft = pts[np.argmin(sm)]
            bottomRight = pts[np.argmax(sm)]
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            pts1 = np.float32([topLeft, topRight, bottomLeft, bottomRight])

            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])

            width = max([w1, w2])
            height = max([h1, h2])

            pts2 = np.float32([[0,0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
            cv2.imshow('trans', result)




cv2.imshow('perspectiveEx', img)
cv2.setMouseCallback('perspectiveEx', perspectiveFunc)
cv2.waitKey()
cv2.destroyAllWindows()