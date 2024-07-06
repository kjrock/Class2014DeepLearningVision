import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img/girl.jpg')
im2 = img[250:700, 250:700]
im2 = cv2.resize(im2, (400, 400))
cv2.imwrite('cut-resize.png', im2)
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()