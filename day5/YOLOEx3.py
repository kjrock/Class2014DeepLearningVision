import cv2
import numpy as np

with open('PreTrained/coco.names','r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(class_names)

np.random.seed(1111)
COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8').tolist()

net = cv2.dnn.readNet('dnn/yolov5n6_v2.onnx')
model = cv2.dnn_DetectionModel(net)

input_size = 640, 640
model.setInputParams(scale=1/255, size=input_size, swapRB=True)

cap = cv2.VideoCapture('data/vtest.avi')

width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def detect(outs, input_size, img_size, confThreshold=0.5, nmsThreshold=0.4):
    ratio_w, ratio_h = width / input_size[0], height / input_size[1]

    labels = []
    class_scores = []
    boxes = []

    outs = outs.squeeze()

    for dectection in outs:
        scores = dectection[5:]
        label = np.argmax(scores)
        class_score = scores[label]
        conf = dectection[4]
        if conf < confThreshold:
            continue

        cx = int(dectection[0] * ratio_w)
        cy = int(dectection[1] * ratio_h)
        w = int(dectection[2] * ratio_w)
        h = int(dectection[3] * ratio_h)

        x = int(cx - w/2)
        y = int(cy - h/2)
        boxes.append([x, y, w, h])
        class_scores.append(float(class_score))
        labels.append(label)

    indices = cv2.dnn.NMSBoxes(boxes, class_scores, confThreshold, nmsThreshold)

    if len(indices) == 0:
        boxes = np.array(boxes)
        class_scores = np.array(class_scores)
        labels = np.array(labels)
    else:
        boxes = np.array(boxes)[indices]
        class_scores = np.array(class_scores)[indices]
        labels = np.array(labels)[indices]

    return labels, class_scores, boxes


while True:
    ret, frame = cap.read()
    if ret:
        outs = model.predict(frame)[0] #1, 25200, 85
        labels, scores, boxes = detect(outs, input_size, (width, height))

        for (label, score, box) in zip(labels, scores, boxes):
            color = COLORS[label]
            text = f'{class_names[label]}:{score:.3f}'
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        break

    cv2.imshow('frame', frame)
    key = cv2.waitKey(5)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
