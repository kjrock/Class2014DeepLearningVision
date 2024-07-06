from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2
import os
import math
import numpy as np
import glob
import matplotlib.pyplot as plt

data_path = 'img/colorize'
data_list = glob.glob(os.path.join(data_path, '*.jpg'))
#print(data_list)

val_n_sample = math.floor(len(data_list) * 0.1)
test_n_sample = math.floor(len(data_list) * 0.1)
train_n_sample = len(data_list) - val_n_sample - test_n_sample

val_lists = data_list[:val_n_sample]
test_lists = data_list[val_n_sample:val_n_sample + test_n_sample]
train_lists = data_list[val_n_sample+test_n_sample:val_n_sample + test_n_sample+train_n_sample]
img_size = 224
print(len(train_lists), len(val_lists), len(test_lists))

def rgb2lab(rgb):
    assert rgb.dtype == 'uint8'
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

def lab2rgb(lab):
    assert lab.dtype == 'uint8'
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def get_lab_from_data_list(data_list):
    x_lab = []
    for f in data_list:
        rgb = img_to_array(load_img(f, target_size=(img_size, img_size))).astype(np.uint8)
        lab = rgb2lab(rgb)
        x_lab.append(lab)
    return np.stack(x_lab)


def generator_with_preprocessing(data_list, batch_size, shuffle=False):
    while True:
        if shuffle:
            np.random.shuffle(data_list)
        for i in range(0, len(data_list), batch_size):
            batch_list = data_list[i: i+batch_size]
            batch_lab = get_lab_from_data_list(batch_list)
            batch_l = batch_lab[:,:,:, 0:1] # L추출
            batch_ab = batch_lab[:,:,:, 1:] # AB추출
            yield  (batch_l, batch_ab)







