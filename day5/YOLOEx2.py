import cv2
import numpy as np

with open('PreTrained/coco.names','r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(class_names)

np.random.seed(1111)
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8').tolist()


net = cv2.dnn.readNet('PreTrained/yolov4.cfg', 'PreTrained/yolov4.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416,416), swapRB=True)

cap = cv2.VideoCapture('data/vtest.avi')

while True:
    ret, frame = cap.read()
    if ret:
        labels, scores, boxes = model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

        for (label, score, box) in zip(labels, scores, boxes):
            #print(label)
            color = COLORS[label % len(COLORS)]
            text = f'{class_names[label]}, prob:{score:.3f}'
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        break

    cv2.imshow('src', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()