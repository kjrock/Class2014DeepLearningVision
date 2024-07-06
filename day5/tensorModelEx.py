from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import mnist

(x_train, y_train), (x_test, y_test) = mnist.load(flatten=True, one_hot=True, normalize=True)

model = tf.keras.Sequential([
    Input(shape=784,),
    Dense(20, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=64)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test acc:',test_acc)

import freeze_graph
freeze_graph.freeze_model(model, 'MNIST_DENSE.pb')