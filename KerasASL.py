# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:29:04 2020

@author: chris
"""

import math
import os
from glob import glob
import cv2
from random import randint, sample
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from multiprocessing import freeze_support, Process

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, layers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2


TARGET_SIZE = (64, 64)
VALIDATION_SPLIT = 0.3

TRAIN_DIR = './asl_alphabet_train'
TEST_DIR = './asl_alphabet_test'
CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
CLASSES.sort()


def get_sample_img(letter):
    img_path = TRAIN_DIR + '/' + letter + '/**'
    path_contents = glob(img_path)
    imgs = sample(path_contents, 1)
    img = cv2.resize(cv2.imread(imgs[0]), TARGET_SIZE)
    
    plt.figure(figsize=(32, 32))
    plt.subplot(121)
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(122)
    plt.imshow(img)
    return img

data_augmentor = ImageDataGenerator(
    samplewise_center=True, 
    samplewise_std_normalization=True,
    horizontal_flip=True,
    rescale=1.0/255,
    validation_split=VALIDATION_SPLIT)

train_generator = data_augmentor.flow_from_directory(
    TRAIN_DIR,
    batch_size=50,
    class_mode='sparse',
    target_size=TARGET_SIZE,
    subset='training')

validation_generator = data_augmentor.flow_from_directory(
    TRAIN_DIR,
    batch_size=50,
    class_mode='sparse',
    target_size=TARGET_SIZE,
    subset='validation')

def findKey(indices, search_value):
    for key, value in indices.items():
        if(value == search_value):
            return key
    return -1


model = Sequential()
model.add(Conv2D(input_shape=(64,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=29, activation="softmax"))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,steps_per_epoch=1218, epochs=10, validation_data=validation_generator, validation_steps=522, use_multiprocessing=True) 

