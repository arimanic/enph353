#!/usr/bin/env python

import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend
from keras.utils import to_categorical

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

## Data engine

# Given a license plate image, cut into 4 images, that each include a single letter or number
# Return a list of cropped images and a list of characters where images(idx) is an image of char(idx)
def cropPlate(plateFile):
    x1, y1 = 50, 83
    x2, y2 = 150, 83
    x3, y3 = 345, 83
    x4, y4 = 440, 83

    width, height = 100, 225-83

    name = plateFile[-8:-4]

    plate = cv2.imread(plateFile)

    img = list()

    img.append(plate[y1:y1+height, x1:x1+width])
    img.append(plate[y2:y2+height, x2:x2+width])
    img.append(plate[y3:y3+height, x3:x3+width])
    img.append(plate[y4:y4+height, x4:x4+width])

    return img, name

# Read all raw data and for each image, cut into individual letters and save to a data folder
def genData():

    # Get all the pictures
    path = join(os.getcwd(), "pictures")
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    # Split them into individual files
    for f in files:
        images, names = cropPlate(f)
        num = np.random.randint(0, high = 100000000, size = 4)
        savePath = join(os.getcwd(), "data")

        for i in range(len(images)):
            cv2.imwrite(join(savePath, "{}_{}.png".format(names[i],num[i])), images[i])


# Given a list of labels, return a 2d array where each row is
# the one hot vector associated with the label given in Y_orig
def convert_to_one_hot(Y_orig):
    # Define a conversion lookup table
    map_from = np.array(sorted(list(set(Y_orig))))
    map_to = np.linspace(0, len(map_from) - 1, len(map_from), dtype = np.uint8)
    one_hot_map = [map_from.T, map_to.T]

    # Populate the list of labels with one hot vectors
    Y = np.zeros((len(Y_orig), len(map_from)))
    for i in range(len(Y_orig)):
        row = np.zeros(len(map_from))

        for hot_idx in range(len(map_from)):
            if Y_orig[i] == map_from[hot_idx]:
                row[map_to[hot_idx]] = 1
                break

        Y[i,:] = row
 
    return Y

def classify():

    # Get all the pictures
    path = join(os.getcwd(), "data")
    dataSet = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    X = list()
    Y = list()

    # Load all images
    for f in dataSet:
        X.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY))
        Y.append(f[23])

    # Convert to the right dimensions for the CNN
    X = np.array(X)
    X = X.reshape(len(dataSet),142, 100,1)
    Y = np.array(Y)
    
    # Normalize X (images) dataset
    X_dataset = X/255.

    # Convert Y dataset to one-hot encoding
    Y_dataset = convert_to_one_hot(Y)

    # Define the layers of the CNN
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(3, (3,3), activation='relu', strides = 8,
                             input_shape=(142, 100, 1)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dense(36, activation='softmax'))

    conv_model.summary()

    conv_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Fit the model to the data
    history_conv = conv_model.fit(X_dataset, Y_dataset, verbose = 1,
                              validation_split=0.2, 
                              epochs=25, 
                              batch_size=32)

    predicted = conv_model.predict(X_dataset)

    labels = [x for x in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    gndTruth = [labels[i] for i in np.argmax(Y_dataset, axis = 1)]
    pred = [labels[i] for i in np.argmax(predicted, axis = 1)]
    matrix = confusion_matrix( gndTruth, pred, labels=labels)
    
    df_cm = pd.DataFrame(matrix, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

    # plt.matshow(matrix)
    # plt.colorbar()
    plt.figure()    

    # Plot results
    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')

    plt.figure()
    plt.plot(history_conv.history['accuracy'])
    plt.plot(history_conv.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()


    

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

#genData()

classify()