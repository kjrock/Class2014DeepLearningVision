import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = np.zeros((200, 400), dtype=np.uint8)
img2 = np.zeros((200, 400), dtype=np.uint8)


img1[:, :200] = 255
img2[100:200, :] = 255

bitAnd = cv2.bitwise_and(img1, img2)
bitOr = cv2.bitwise_or(img1, img2)
bitXor = cv2.bitwise_xor(img1, img2)
bitNot = cv2.bitwise_not(img1)

imgs = {'img1':img1, 'img2':img2, 'and':bitAnd, 'or':bitOr, 'xor':bitXor, 'not(img1)':bitNot}

plt.figure(figsize=(10, 10))
for i, (title, im) in enumerate(imgs.items()):
    plt.subplot(3,2, i+1)
    plt.title(title)
    plt.imshow(im, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()