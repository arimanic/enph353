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
import tensorflow as tf

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Comment these out when running in ROS
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd

from constants import *

class Classifier:
    def __init__(self, outLabels):
        self.conv_model = models.Sequential()
        self.conv_model.add(layers.Conv2D(16, (3,3), activation='relu', strides = 1, padding = 'same',
                                input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)))
        self.conv_model.add(layers.MaxPooling2D((2, 2)))
        self.conv_model.add(layers.Conv2D(32,(3,3), activation='relu', padding = 'same'))
        self.conv_model.add(layers.Conv2D(32,(5,5), activation='relu', padding = 'same'))
        self.conv_model.add(layers.MaxPooling2D((2, 2)))
        self.conv_model.add(layers.Conv2D(32,(7,7), activation='relu'))


        self.conv_model.add(layers.Flatten())
        self.conv_model.add(layers.Dropout(0.5))
        
        self.conv_model.add(layers.Dense(len(outLabels)*5, activation='relu'))
        self.conv_model.add(layers.Dense(len(outLabels)*5, activation='relu'))

        self.conv_model.add(layers.Dense(len(outLabels), activation='softmax'))

        self.conv_model.summary()

        self.conv_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        ## Labels
        self.labels = outLabels

        self.sess = tf.Session(target='', graph=None, config=None)
        self.graph = tf.get_default_graph()
        backend.set_session(self.sess)

    def loadLayers(self,path):
        self.conv_model = models.loadModel(path)

    def saveLayers(self,path):
        backend.set_session(self.sess)
        self.conv_model.save(path)

    def train(self, dataPath, savePath):
        # Get all the pictures
        path = dataPath
        dataSet = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        X = list()
        Y = list()

        # Load all images
        for f in dataSet:
            X.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY))

            # Find the label
            idx = f[::-1].find("_")
            Y.append(f[-(idx+2)])

        # Convert to the right dimensions for the CNN
        X = np.array(X)
        X = X.reshape(len(dataSet),INPUT_HEIGHT, INPUT_WIDTH,1)
        Y = np.array(Y)
        
        # Normalize X (images) dataset
        X_dataset = X/255.

        # Convert Y dataset to one-hot encoding
        Y_dataset = convert_to_one_hot(Y)

        # Create a callback to save a checkpoint file after each epoch
        cp_callback = ModelCheckpoint(filepath=savePath,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True,
                                        save_weights_only=True,
                                        verbose=1)

        es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.001)


        # Fit the model to the data
        self.history_conv = self.conv_model.fit(X_dataset, Y_dataset, verbose = 1,
                                validation_split=0.2, 
                                epochs=100, 
                                batch_size=1000,
                                workers=8,
                                callbacks=[cp_callback, es_callback])

        self.X_dataset = X_dataset
        self.Y_dataset = Y_dataset

    def showMetrics(self):
        with self.graph.as_default():
            backend.set_session(self.sess)
            predicted = self.conv_model.predict(self.X_dataset)

            labels = [x for x in self.labels]
            gndTruth = [labels[i] for i in np.argmax(self.Y_dataset, axis = 1)]
            pred = [labels[i] for i in np.argmax(predicted, axis = 1)]
            matrix = confusion_matrix( gndTruth, pred, labels=labels)
            
            df_cm = pd.DataFrame(matrix, index = labels,
                        columns = labels)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True)


            plt.figure()    

            # Plot results
            plt.plot(self.history_conv.history['loss'])
            plt.plot(self.history_conv.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss'], loc='upper left')

            plt.figure()
            plt.plot(self.history_conv.history['accuracy'])
            plt.plot(self.history_conv.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy (%)')
            plt.xlabel('epoch')
            plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
            plt.show()

    def loadWeights(self, loadPath):
        self.conv_model.load_weights(loadPath)
        self.conv_model._make_predict_function()

    def predict(self, data):
        with self.graph.as_default():
            backend.set_session(self.sess)
            prediction = self.conv_model.predict(data)
            characters = [self.labels[i] for i in np.argmax(prediction, axis = 1)]

            return "".join(characters)



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