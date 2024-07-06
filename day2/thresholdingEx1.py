import matplotlib.pyplot as plt
import cv2

img = cv2.imread('data/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

ret, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, t_bin_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, t_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, t_to_zero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, t_to_zero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'original':img, 'BINARY':t_bin, 'BINARY_INV':t_bin_inv,
        'TRUNC':t_trunc, 'TOZERO':t_to_zero, 'TOZERO_INV':t_to_zero_inv}

for i, (title, bimg) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(title)
    plt.imshow(bimg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
