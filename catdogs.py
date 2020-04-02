#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
directory = os.getcwd()+"/PetImages"
categories = 'Dog Cat'.split()
imgSize = 60
trainingSet = []
def createTrainingSet():
    for category in categories:
        path = os.path.join(directory,category)
        for imgIndex in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path,imgIndex),cv2.IMREAD_GRAYSCALE)
                resizedImg = cv2.resize(img,(imgSize,imgSize))
                trainingSet.append([resizedImg,categories.index(category)])
            except Exception as e:
                pass
createTrainingSet()
import random
random.shuffle(trainingSet)
inputImageFeatues = []
target = []
for features,label in trainingSet:
    inputImageFeatues.append(features)
    target.append(label)
inputArray = np.array(inputImageFeatues).reshape(-1,imgSize,imgSize,1)
targets = np.array(target)
np.savez('inputArray.npz',inputForArray = inputArray, targetForArray = targets) #you dont want to perform these calculations everty time
data = np.load('inputArray.npz')
inputs = data['inputForArray']
targetsFinal = data['targetForArray']
inputScaled = inputs/255.0
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(2,2), activation='relu', input_shape = inputScaled.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64,(2,2), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')  
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(inputScaled, targetsFinal, epochs = 5, batch_size = 32 ,validation_split = 0.1)
model.save('catdogsFirstModel.model')
