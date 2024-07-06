import cv2
import numpy as np

net = cv2.dnn.readNet('./PreTrained/resnet152-v2-7.onnx')

image_name = ['./data/elephant.jpg', './data/eagle.jpg', './data/dog.jpg']
src = cv2.imread(image_name[0])

def preprocessing(img):
    x = img.copy()
    x = cv2.resize(x, dsize=(224,224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    x = x / 255

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x - mean) / std

    return np.float32(x)


img = preprocessing(src)
blob = cv2.dnn.blobFromImage(img) # 1, 3, 224, 224
net.setInput(blob)
out = net.forward()

out = out.flatten() # 1000
top5 = out.argsort()[-5:][::-1]
top1 = top5[0]
print('top1:', top1)
print('top5:', top5)

import json
with open('./PreTrained/imagenet_labels.json', 'r') as f:
    imagenet_labels = json.load(f)
print('top1 prediction:', imagenet_labels[top1], out[top1])
print('top5 prediction:', [imagenet_labels[i] for i in top5])