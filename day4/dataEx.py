import matplotlib.pyplot as plt
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_x, train_y),(test_x, test_y) = fashion_mnist.load_data()

print(train_x.shape, train_y.shape)

train_x = train_x.reshape(-1,28,28,1)

plt.figure(figsize=(10,10))
for c in range(16):
    plt.subplot(4,4,c+1)
    plt.imshow(train_x[c].reshape(28,28), cmap='gray')
plt.show()