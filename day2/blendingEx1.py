import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('data/wing_wall.jpg')
img2 = cv2.imread('data/yate.jpg')
print(img1.shape)
print(img2.shape)

img3 = img1 + img2
img4 = cv2.add(img1, img2)

imgs = {'img1':img1, 'img2':img2, 'img1 + img2':img3, 'cv2.add(img1, img2)':img4}

plt.figure(figsize=(10, 10))
for i, (title, im) in enumerate(imgs.items()):
    plt.subplot(2,2, i+1)
    plt.title(title)
    plt.imshow(im[:,:,::-1])
    plt.xticks([])
    plt.yticks([])
plt.show()