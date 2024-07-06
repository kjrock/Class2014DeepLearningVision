from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

tf.random.set_seed(3)

data = np.loadtxt('data/ThoraricSurgery.csv', delimiter=',')
print(data.shape)

x_data = data[:, 0:17]
y_data = data[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

result = model.fit(x_data, y_data, epochs=100, batch_size=10)

print(result.history['loss'])
print(result.history['accuracy'])