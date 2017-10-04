import modelnet40
import batchGenerator
#"""
#HW1 Part II
#"""
#
import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import numpy
#"""
#from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Input,concatenate,Maximum,Conv2D
from keras.models import Model
from keras import models
from keras.layers.normalization import BatchNormalization
STOP_LAYER=20
if __name__ == '__main__':
    
    (g, dataset_size) = modelnet40.modelnet40_generator('test')
    print('Loading first element from dataset')
    (x1, y1) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
####################################################
    sharedResnet = applications.resnet50.ResNet50(include_top=False,input_shape=x1.shape[1:])
    sharedResnet=models.Model(sharedResnet.input, sharedResnet.layers[STOP_LAYER].output)
    Input_instances=[Input(shape=x1.shape[1:]) for i in range(12)]
    resnet_instances=[sharedResnet(Input_instances[i]) for i in range(12)]
#    merged_vector = concatenate(resnet_instances, axis=-1)
    base_model=Maximum()(resnet_instances)
    x = base_model
    x=Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x=BatchNormalization()(x)
    x= Flatten(input_shape=x.shape[1:])(x)
    predictions= Dense(40, activation='softmax')(x)
    whole_model = Model(Input_instances, outputs=predictions)
#    print (resnet.layers)
    p=int(0.7*len(whole_model.layers))
    print(len(whole_model.layers), p)
    for layer in whole_model.layers[:p]:
    	layer.trainable = False
   
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batchSize=32
    augment=True
    single=False
    RangeRotation=True
    whole_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    (train_generator, tSize)= modelnet40.modelnet40_generator('train', src_dir=modelnet40.DEFAULT_SRCDIR, single=single, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (validation_generator,vSize)=modelnet40.modelnet40_generator('test', src_dir=modelnet40.DEFAULT_SRCDIR, single=single, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True,  verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (batchTrainGen,btSize)=batchGenerator.batchGenerator(train_generator,batchSize,tSize,single=single,augment=augment, RangeRotation= RangeRotation)
    (batchValidGen,bvSize)=batchGenerator.batchGenerator(validation_generator,batchSize,vSize,single=single,augment=augment, RangeRotation= RangeRotation)
    whole_model.fit_generator(batchTrainGen, samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=batchValidGen,nb_val_samples=nb_validation_samples) 
