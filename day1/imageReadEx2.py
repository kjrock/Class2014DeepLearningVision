import cv2
import urllib.request as req

url = 'http://uta.pw/shodou/img/28/214.png'
req.urlretrieve(url, 'img/test.png')
img = cv2.imread('img/test.png')
print(img)

import matplotlib.pyplot as plt

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(img)

plt.show()