import cv2
import numpy as np

img = cv2.imread('data/abnormal.jpg', cv2.IMREAD_GRAYSCALE)
img_f = img.astype(np.float32)

img_norm = ((img_f - img_f.min())) * 255 / (img_f.max() - img_f.min())
img_norm = img_norm.astype(np.uint8)

img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


hist = cv2.calcHist([img], [0], None, [256], [0,255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0,255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0,255])

cv2.imshow('before', img)
cv2.imshow('img_norm', img_norm)
cv2.imshow('img_norm2', img_norm2)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
imgs = {'before':hist, 'img_norm':hist_norm, 'img_norm2':hist_norm2}

for i, (title, im) in enumerate(imgs.items()):
    plt.subplot(1,3, i+1)
    plt.title(title)
    plt.plot(im)
plt.tight_layout()
plt.show()