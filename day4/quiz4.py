# 1. CIFAR-10 데이터를 이용하여 아래에 조건에 맞게 모델을 구성하시오
#   ㄱ.10가지 물체를 구별할 수 있는 cnn 모델을 생성한다.
#   ㄴ. 데이터 아래의 방식을 다운로드 받는다.
#    from tensorflow.keras.datasets import cifar10
#    (X_train, y_train), (X_test, y_test) = cifar10.load_data() #size: (32 * 32)
#    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#   ㄷ.제공한 이미지 파일을 학습된 모델을 이용하여 예측한다.


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #size: (32 * 32)
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

x_train = x_train.reshape(-1, 32, 32, 3) / 255.
x_test = x_test.reshape(-1, 32, 32, 3) / 255.

model = Sequential([
    Conv2D(input_shape=(32,32,3), kernel_size=(3,3), filters=32, padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=256, activation='relu'),
    Dropout(0.5),
    Dense(units=10, activation='softmax')
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.25, batch_size=100)
print(model.evaluate(x_test, y_test))