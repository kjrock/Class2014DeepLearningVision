import cv2

img_file = 'img/girl.jpg'
img = cv2.imread(img_file)
print(img)
print(type(img))
print(img.shape)

if img is not None:
    print('show')
    cv2.imshow('IMG', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file')