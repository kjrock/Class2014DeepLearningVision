import cv2
import matplotlib.pyplot as plt

b_size = 7
C = 3

img = cv2.imread('data/sudoku.png', cv2.IMREAD_GRAYSCALE)

t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, C)

imgs = {'original':img, f'otsu_{t}':t_otsu, 'adapted_mean':th2, 'adapted_gaussian':th3}

plt.figure(figsize=(10, 10))
for i, (title, bimg) in enumerate(imgs.items()):
    plt.subplot(2,2, i+1)
    plt.title(title)
    plt.imshow(bimg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()