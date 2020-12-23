# -*- coding: utf-8 -*-
"""

@author: Taha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:18:30 2018

@author: FaraDars
"""
import keras
import tensorflow
from keras.datasets import mnist
from plot_history import plot_history

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

X_train = train_images.reshape(60000, 28, 28, 1)
X_test = test_images.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

from keras.utils import np_utils
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

#==================================================
# Creating our model
from keras.models import Model
from keras import layers
import keras

myInput = layers.Input(shape=(28,28,1))
conv1 = layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(myInput)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(conv1)
flat = layers.Flatten()(conv2)
out_layer = layers.Dense(10, activation='softmax')(flat)

myModel = Model(myInput, out_layer)

myModel.summary()
myModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#==================================================
# Train our model
network_history = myModel.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
plot_history(network_history)

# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)

# Change layers config
myModel.layers[0].name = 'Layer_0'
myModel.layers[0].trainable = False
myModel.layers[0].get_config()
