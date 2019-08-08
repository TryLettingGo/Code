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
validation_data_dir = train_data_dir
nb_train_samples = 17250
nb_validation_samples = 100
batch_size = 7
epochs = 2

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
        rotation_range=30,
        validation_split = 0.2)

'''test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = "nearest",
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range=0.3,
        rotation_range=30)'''

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        class_mode = "categorical",
        subset = 'training')

validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = "categorical",
        subset = 'validation')

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
model_final.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        callbacks = [checkpoint, early])


#test an image
def test_image(filename):
    
    img_path = "C:/Users/MSI/Documents/Github/FacialRecAnalysis/Data/"
    load = image.load_img(img_path + filename, target_size = (256, 256))
    array = image.img_to_array(load)
    exd = np.expand_dims(array, axis = 0)
    test = keras.applications.mobilenet.preprocess_input(exd)
    pred = model_final.predict(test)
    
    return pred

bmdiana = test_image("image.jpg")
karma = test_image("image2.jpg")
ezreal = test_image("image3.jpg")
frezreal = test_image("image4.jpg")
diana = test_image("image5.jpg")
camille = test_image("image6.jpg")
hugh_jackman = test_image("image7.jpg")


print("Your neural net is perfect, there's nothing to worry about.")