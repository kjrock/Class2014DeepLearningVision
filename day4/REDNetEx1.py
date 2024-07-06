import tensorflow as tf
import numpy as np
import pathlib

image_root = pathlib.Path('content/images')
all_image_paths = list(image_root.glob('*/*'))
print(all_image_paths)

train_path, valid_path, test_path = [], [], []

for image_path in all_image_paths:
    if str(image_path).split('.')[-1] != 'jpg':
        continue
    if str(image_path).split('\\')[-2] == 'train':
        train_path.append(str(image_path))
    elif str(image_path).split('\\')[-2] == 'val':
        valid_path.append(str(image_path))
    else:
        test_path.append(str(image_path))

def get_hr_and_lr(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    hr = tf.image.random_crop(img, [50,50, 3])
    lr = tf.image.resize(hr, [25, 25])
    lr = tf.image.resize(lr, [50, 50])
    return lr, hr

train_dataset = tf.data.Dataset.list_files(train_path)
train_dataset = train_dataset.map(get_hr_and_lr)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(16)

valid_dataset = tf.data.Dataset.list_files(valid_path)
valid_dataset = valid_dataset.map(get_hr_and_lr)
valid_dataset = valid_dataset.repeat()
valid_dataset = valid_dataset.batch(1)

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation

def REDNet(num_layers):
    conv_layers = []
    deconv_layers = []
    residual_layers = []

    inputs = Input(shape=(None, None, 3))
    conv_layers.append(Conv2D(3, kernel_size=3, padding='same', activation='relu'))
    for i in range(num_layers - 1):
        conv_layers.append(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        deconv_layers.append(Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu'))
    deconv_layers.append(Conv2DTranspose(3, kernel_size=3, padding='same'))

    x = conv_layers[0](inputs)

    for i in range(num_layers - 1):
        x = conv_layers[i+1](x)
        if i % 2 == 0:
            residual_layers.append(x)

    for i in range(num_layers - 1):
        if i % 2 == 1:
            x = Add()([x, residual_layers.pop()])
            x = Activation('relu')(x)
        x = deconv_layers[i](x)
    x = deconv_layers[-1](x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

model = REDNet(15)
model.summary()

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

model.compile(loss='mse',
              optimizer=tf.optimizers.Adam(0.0001),
              metrics=[psnr_metric])

result = model.fit_generator(train_dataset,
                             epochs=70,
                             steps_per_epoch=len(train_path)//16,
                             validation_data=valid_dataset,
                             validation_steps=len(valid_path))

import matplotlib.pyplot as plt
plt.plot(result.history['psnr_metric'], 'b-', label='psnr')
plt.plot(result.history['val_psnr_metric'], 'r--', label='val_psnr')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

img = tf.io.read_file(test_path[0])
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

lr = tf.image.resize(hr, [hr.shape[0]//2, hr.shape[1]//2])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])
predict_hr = model.predict(np.expand_dims(lr, axis=0))
print(tf.image.psnr(np.squeeze(predict_hr, axis=0), hr, max_val=1.0))

plt.figure(figsize=(8, 16))
plt.subplot(3,1,1)
plt.imshow(hr)
plt.title('original')

plt.subplot(3,1,2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(3,1,3)
plt.imshow(np.squeeze(predict_hr, axis=0))
plt.title('predict_hr')
plt.show()












