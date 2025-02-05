from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

if not os.path.exists('./gan_images'):
    os.makedirs('./gan_images')

tf.random.set_seed(3)

generator = Sequential([
    Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)),
    BatchNormalization(),
    Reshape((7,7,128)),
    UpSampling2D(),
    Conv2D(64, kernel_size=5, padding='same'),
    BatchNormalization(),
    Activation(LeakyReLU(0.2)),
    UpSampling2D(),
    Conv2D(1, kernel_size=5, padding='same', activation='tanh')
])

generator.summary()


discriminator = Sequential([
    Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Conv2D(128, kernel_size=5, strides=2, padding='same'),
    Activation(LeakyReLU(0.2)),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                      optimizer='adam')

discriminator.trainable = False
ginput = Input(shape=(100,))
ganEx = discriminator(generator(ginput))
gan = Model(ginput, ganEx)
gan.compile(loss='binary_crossentropy',
            optimizer='adam')

def gan_train(epoch, batch_size, saving_interval):
    (x_train, _),(_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_train = (x_train - 127.5) / 127.5

    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for i in range(epoch):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        d_loss_real = discriminator.train_on_batch(imgs, true)

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        g_loss = gan.train_on_batch(noise, true)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print(f'epoch{i}, d_loss:{d_loss:.4f}, g_loss:{g_loss:.4f}')

        if i % saving_interval == 0:
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axes = plt.subplots(5, 5)
            count = 0
            for j in range(5):
                for k in range(5):
                    axes[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axes[j, k].axis('off')
                    count += 1
            fig.savefig('gan_images/gan_fimage_%d.png'%i)


gan_train(3000, 32, 100)












