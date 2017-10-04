import modelnet40
import batchGenerator
################################################
import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import numpy
"""
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

if __name__ == '__main__':
    (g, dataset_size) = modelnet40.modelnet40_generator('test')
    (x1, y1) = g.__next__()     
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batchSize=32
    augment=True
    RangeRotation=True
    base_model=applications.ResNet50(weights='imagenet',input_shape=x1.shape[1:], include_top=False)
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    x = base_model.output
    x= Flatten(input_shape=base_model.output_shape[1:])(x)
    predictions= Dense(40, activation='softmax')(x)
    whole_model = Model(inputs=base_model.input, outputs=predictions)
    p=int(0.7*len(whole_model.layers))
    print(len(whole_model.layers), p)
    for layer in whole_model.layers[:p]:
    	layer.trainable = False
    whole_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    (train_generator, tSize)= modelnet40.modelnet40_generator('train', src_dir=modelnet40.DEFAULT_SRCDIR, single=True, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (validation_generator,vSize)=modelnet40.modelnet40_generator('test', src_dir=modelnet40.DEFAULT_SRCDIR, single=True, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True,  verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (batchTrainGen,btSize)=batchGenerator.batchGenerator(train_generator,batchSize,tSize,augment=augment,RangeRotation=RangeRotation)
    (batchValidGen,bvSize)=batchGenerator.batchGenerator(validation_generator,batchSize,vSize,augment=augment, RangeRotation=RangeRotation)
    whole_model.fit_generator(batchTrainGen, samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=batchValidGen,nb_val_samples=nb_validation_samples)
 
