import modelnet40

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
#DEFAULT_SRCDIR = 'modelnet40'
#DEFAULT_TARGET_SIZE = (224, 224)        # Input size for ResNet-50
#nclasses = 40
#nviews = 12
#
#def subdirs(dirname):
#    return [x for x in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, x))]
#
#def read_image(filename, target_size, preprocess=None):
##    print('read_image:', filename)
#    x = image.load_img(filename, target_size=target_size)
#    x = image.img_to_array(x)
#    x = np.expand_dims(x, axis=0)
#    if preprocess is not None:
#        x = preprocess(x)
#    #print('read_image, shape:', x.shape)
#    return x
#
#def modelnet40_filenames(subset, src_dir=DEFAULT_SRCDIR):
#    """
#    List of models for ModelNet-40.
#    
#    Each model is a pair (class_index, filename_L).
#    Here filename_L is a 12 length list of image filenames of views of the model.
#    """
#    src_dir = os.path.join(src_dir, 'classes')
#    classes = sorted(subdirs(src_dir))
#    ans = []
#    for (icls, cls) in enumerate(classes):
#        subset_dir = os.path.join(src_dir, cls, subset)
#        model_dirs = subdirs(subset_dir)
#        for model_dir in model_dirs:
#            filenames = glob.glob(os.path.join(src_dir, cls, subset, model_dir, '*.png'))
#            ans.append((icls, filenames))
#    return ans
#
#def modelnet40_generator(subset, src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input):
#    """
#    A generator that returns images and classes from ModelNet-40 in size one batches.
#    
#    Returns (g, dataset_size), where g is the generator and dataset_size is the number of elements in the dataset.
#    
#    The generator yields by default an infinite number of elements which each have the form (x, y), where x is
#    an input for supervised training and y is an output. If single is True (single view mode) then x is a 4D numpy
#    array with shape 1 x h x w x 3, representing an input image, and y is a 2D tensor of shape 1 x nclasses, where
#    nclasses is the number of classes in ModelNet-40 (defined as the global nclasses, equal to 40). If single is False
#    (multiple view mode) then x is a list of arrays representing different views of the same model: the list has length
#    nviews (defined as the global nviews, equal to 12): each view is an numpy array of an image with shape 1 x h x w x 3.
#    
#    Arguments:
#    - subset:       Either 'train' or 'test'
#    - src_dir:      The ModelNet-40 directory.
#    - single:       If true, return one image and class at a time in the format (img, cls).
#                    If false, return a list of (img, cls) for all (12) views.
#    - repeats:      Number of times to repeat the dataset. If None, repeat forever.
#    - shuffle:      If true, randomly shuffle the dataset.
#    - verbose:      Print information about loading: verbose=0: no info, 1 is some info, verbose=2 is more info.
#    - frac:         Fraction of dataset to load (use frac < 1.0 for quick tests).
#    - class_array:  If true, return class as a 1-hot vector array.
#    - preprocess:   Preprocessing function from a Keras model to be called on each image (numpy array) after being read.
#                    (or None, the default, for no preprocessing).
#    """
#    viewL = modelnet40_filenames(subset, src_dir)
#    def generator_func():
#        repeat = 0
#        while repeats is None or repeat < repeats:
#            if shuffle:
#                random.shuffle(viewL)
#            for (i, view) in enumerate(viewL[:int(len(viewL)*frac)]):
#                if verbose == 1 and i % 100 == 0:
#                    print('Loading %s data: %.1f%%' % (subset, i*100.0/(len(viewL)*frac)))
#                (cls, view) = view
#                if verbose == 2:
#                    print('Loading data point %d, cls = %d' % (i, cls))
#                if class_array:
#                    cls_array = numpy.zeros((1, nclasses), 'float32')
#                    cls_array[0, cls] = 1.0
#                    cls = cls_array
#                if single:
#                    filename = random.choice(view)
#                    yield (read_image(filename, target_size, preprocess), cls)
#                else:
#                    yield ([read_image(view_elem, target_size, preprocess) for view_elem in view], cls)
#            repeat += 1 
#    return (generator_func(), len(viewL))
def batchGenerator( SingleGenerator, batchSize, dataSetSize,single=True,augment=True):
	def nestedGen():
		while True:
			for i in range(batchSize):
				(a,b)=SingleGenerator.__next__() 
#				print("len a is",len(a))
#				print(a[0])
				if(augment==True and single==True):
					a[0]=image.random_rotation(a[0],45, row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.)
				if(augment==True and single==False):
					for j in range(len(a)):
						a[j][0]=image.random_rotation(a[j][0],45, row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.)
				
						
					#random horizontal flip along columns
				if(augment==True and single==True):
					rnd=np.random.uniform(0,1)
					if(rnd<0.5):
						a=image.flip_axis(a, 2)
				if(augment==True and single==False):
					for j in range(len(a)):
						rnd=np.random.uniform(0,1)
						if(rnd<0.5):
							a[j]=image.flip_axis(a[j], 2)
				if(i==0):
					(array_a,array_b)=(a,b)
				else:
	#				print("hi")
					if(single==True):
						array_a=np.append(array_a,a,axis=0)
						array_b=np.append(array_b,b,axis=0)
					else:
#						print(len(a))
						for j in range(len(a)):
							array_a[j]=np.append(array_a[j],a[j],axis=0)
						array_b=np.append(array_b,b,axis=0)
#					print(a.shape, b.shape)
	#						print(array_a[j].shape)
			yield (array_a,array_b)
	return (nestedGen(),dataSetSize )
STOP_LAYER=20
if __name__ == '__main__':
    
    (g, dataset_size) = modelnet40.modelnet40_generator('test')
    print('Loading first element from dataset')
    (x1, y1) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
    print(x1, y1)
    print('Loading second element from dataset')
    (x2, y2) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
    print(x2, y2)
    print('Done loading')
    print(x2.shape)
    (Tgen,Tsize)=batchGenerator(g,5,dataset_size)
    print(Tgen.__next__()[0].shape )

####################################################
    sharedResnet = applications.resnet50.ResNet50(include_top=False,input_shape=x2.shape[1:])
    sharedResnet=models.Model(sharedResnet.input, sharedResnet.layers[STOP_LAYER].output)
    Input_instances=[Input(shape=x2.shape[1:]) for i in range(12)]
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
    batchSize=5
    augment=True
    single=False
    whole_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    (train_generator, tSize)= modelnet40.modelnet40_generator('train', src_dir=modelnet40.DEFAULT_SRCDIR, single=single, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (validation_generator,vSize)=modelnet40.modelnet40_generator('test', src_dir=modelnet40.DEFAULT_SRCDIR, single=single, target_size=modelnet40.DEFAULT_TARGET_SIZE, repeats=None, shuffle=True,  verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    (batchTrainGen,btSize)=batchGenerator(train_generator,batchSize,tSize,single=single,augment=augment)
    (batchValidGen,bvSize)=batchGenerator(validation_generator,batchSize,vSize,single=single,augment=augment)
    whole_model.fit_generator(batchTrainGen, samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=batchValidGen,nb_val_samples=nb_validation_samples) 
