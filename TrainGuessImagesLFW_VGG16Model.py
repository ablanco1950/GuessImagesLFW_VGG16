# -*- coding: utf-8 -*-
"""
Adapted from
 https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
 https://stackoverflow.com/questions/52575271/keras-vgg16-same-model-different-approach-gave-different-result

by Alfonso Blanco García , March 2023
"""

######################################################################
# PARAMETERS
######################################################################

batch_size = 128
epochs = 30
######################################################################
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import time

## image path
train_data_dir = 'lfw5\\lfw5train'
validation_data_dir = 'lfw5\\lfw5valid'
## other
img_width, img_height = 250, 250
#nb_train_samples = 100
#nb_validation_samples = 800
top_epochs = 50
fit_epochs = 50
batch_size = 24
nb_classes = 15
nb_epoch = 50

#　start measurement
start = time.time()

# import vgg16 model
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 =tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# creating an FC layer
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))
top_model.summary()
# bound VGG 16 and FC layer
vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

print(vgg_model.layers[:15])
# prevent re-learning of the layer before the last convolution layer
for layer in vgg_model.layers[:15]:
    layer.trainable = False
vgg_model.summary()
# create model
vgg_model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)

# Setting learning data
train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

history = vgg_model.fit_generator(
        train_generator,
        #steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=validation_generator,
        #validation_steps=nb_validation_samples
)
vgg_model.save("ModelGuessImages_LFW_VGG16.h5")
