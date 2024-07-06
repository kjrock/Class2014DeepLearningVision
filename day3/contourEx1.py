import cv2

img = cv2.imread('img/shapes.png')
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

contour, hierachy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contour2, hierachy2 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour2, hierachy2 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
# contour2, hierachy2 = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

print(f'도형 갯 수 : {len(contour)}, {len(contour2)}')

cv2.drawContours(img, contour, -1, (0, 255, 0), 4)
cv2.drawContours(img2, contour2, -1, (0, 255, 0), 4)

for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (0,0,255), -1)

for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 1, (0,0,255), -1)

cv2.imshow('contour1', img)
cv2.imshow('contour2', img2)

cv2.waitKey()
cv2.destroyAllWindows()