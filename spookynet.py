# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:03:46 2019

@author: MSI
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications import imagenet_utils
import keras
import os
import numpy as np

img_width = 256
img_height = 256 
train_data_dir = "D:/query_data/facialrec/Images"
validation_data_dir = "data/val"
nb_train_samples = 9000
nb_validation_samples = 4
batch_size = 4
epochs = 5

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:5]:
    layer.trainable = False
    
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)

test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        class_mode = "categorical")
'''
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        class_mode = "categorical")
'''
# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
model_final.fit_generator(
        train_generator,
        samples_per_epoch = nb_train_samples,
        epochs = epochs,
        #validation_data = validation_generator,
        #nb_val_samples = nb_validation_samples,
        callbacks = [checkpoint, early])


#test an image
img_path = "C:/Users/MSI/Documents/Github/FacialRecAnalysis/Data/"

diana = image.load_img(img_path + "image.jpg", target_size = (256, 256))
karma = image.load_img(img_path + "image2.jpg", target_size = (256, 256))
ezreal = image.load_img(img_path + "image3.jpg", target_size = (256, 256))

array1 = image.img_to_array(diana)
array2 = image.img_to_array(karma)
array3 = image.img_to_array(ezreal)
exd1 = np.expand_dims(array1, axis = 0)
exd2 = np.expand_dims(array2, axis = 0)
exd3 = np.expand_dims(array3, axis = 0)
diana_test = keras.applications.mobilenet.preprocess_input(exd1)
karma_test = keras.applications.mobilenet.preprocess_input(exd2)
ezreal_test = keras.applications.mobilenet.preprocess_input(exd3)

pred1 = model_final.predict(diana_test)
pred2 = model_final.predict(karma_test)
pred3 = model_final.predict(ezreal_test)
result1 = imagenet_utils.decode_predictions(pred1)
result2 = imagenet_utils.decode_predictions(pred2)
result3 = imagenet_utils.decode_predictions(pred3)

test1 = model_final.predict(diana)
test2 = model_final.predict(karma)
test3 = model_final.predict(ezreal)