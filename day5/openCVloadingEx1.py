import cv2
import numpy as np

net = cv2.dnn.readNet('dnn/MNIST_DENSE.pb')

import mnist

(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, normalize=True)

def calcAccuarcy(y_true, y_pred, percent=True):
    N = y_true.shape[0]
    accuracy = np.sum(y_pred == y_true) / N
    if percent:
        accuracy *= 100
    return accuracy

n = x_train.shape[1] # 784
blob = x_train.reshape(-1, 1, 1, n)
#print(blob.shape)

net.setInput(blob)
y_out = net.forward()
y_pred = np.argmax(y_out, axis=1)
train_accuracy = calcAccuarcy(y_train, y_pred)
print('train accuracy:',train_accuracy)