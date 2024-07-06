import cv2

with open('PreTrained/coco.names','r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(class_names)

COLORS = [(0,0,255),(0,255,0),(255,0,0),
          (255,255,255),(0,255,255),(255,0,255),(255,128,255)]

src = cv2.imread('data/person.jpg')

net = cv2.dnn.readNet('PreTrained/yolov4.cfg', 'PreTrained/yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416,416), swapRB=True)

labels, scores, boxes = model.detect(src, confThreshold=0.5, nmsThreshold=0.4)

for (label, score, box) in zip(labels, scores, boxes):
    #print(label)
    color = COLORS[label % len(COLORS)]
    text = f'{class_names[label]}, prob:{score:.3f}'
    cv2.rectangle(src, box, color, 2)
    cv2.putText(src, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()