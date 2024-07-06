import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE)

_, t_100 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('otus threshold:', t)

imgs = {'original':img, 't_100':t_100, f'otsu_{t}':t_otsu}

plt.figure(figsize=(12, 6))
for i, (title, bimg) in enumerate(imgs.items()):
    plt.subplot(1,3, i+1)
    plt.title(title)
    plt.imshow(bimg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()