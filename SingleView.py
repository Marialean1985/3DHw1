
"""
Multiview (MVCNN) ModelNet-40 dataset for Keras
"""

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
DEFAULT_SRCDIR = 'modelnet40'
DEFAULT_TARGET_SIZE = (224, 224)        # Input size for ResNet-50
nclasses = 40
nviews = 12

def subdirs(dirname):
    return [x for x in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, x))]

def read_image(filename, target_size, preprocess=None):
#    print('read_image:', filename)
    x = image.load_img(filename, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    if preprocess is not None:
        x = preprocess(x)
    #print('read_image, shape:', x.shape)
    return x

def modelnet40_filenames(subset, src_dir=DEFAULT_SRCDIR):
    """
    List of models for ModelNet-40.
    
    Each model is a pair (class_index, filename_L).
    Here filename_L is a 12 length list of image filenames of views of the model.
    """
    src_dir = os.path.join(src_dir, 'classes')
    classes = sorted(subdirs(src_dir))
    ans = []
    for (icls, cls) in enumerate(classes):
        subset_dir = os.path.join(src_dir, cls, subset)
        model_dirs = subdirs(subset_dir)
        for model_dir in model_dirs:
            filenames = glob.glob(os.path.join(src_dir, cls, subset, model_dir, '*.png'))
            ans.append((icls, filenames))
    return ans

def modelnet40_generator(subset, src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input):
    """
    A generator that returns images and classes from ModelNet-40 in size one batches.
    
    Returns (g, dataset_size), where g is the generator and dataset_size is the number of elements in the dataset.
    
    The generator yields by default an infinite number of elements which each have the form (x, y), where x is
    an input for supervised training and y is an output. If single is True (single view mode) then x is a 4D numpy
    array with shape 1 x h x w x 3, representing an input image, and y is a 2D tensor of shape 1 x nclasses, where
    nclasses is the number of classes in ModelNet-40 (defined as the global nclasses, equal to 40). If single is False
    (multiple view mode) then x is a list of arrays representing different views of the same model: the list has length
    nviews (defined as the global nviews, equal to 12): each view is an numpy array of an image with shape 1 x h x w x 3.
    
    Arguments:
    - subset:       Either 'train' or 'test'
    - src_dir:      The ModelNet-40 directory.
    - single:       If true, return one image and class at a time in the format (img, cls).
                    If false, return a list of (img, cls) for all (12) views.
    - repeats:      Number of times to repeat the dataset. If None, repeat forever.
    - shuffle:      If true, randomly shuffle the dataset.
    - verbose:      Print information about loading: verbose=0: no info, 1 is some info, verbose=2 is more info.
    - frac:         Fraction of dataset to load (use frac < 1.0 for quick tests).
    - class_array:  If true, return class as a 1-hot vector array.
    - preprocess:   Preprocessing function from a Keras model to be called on each image (numpy array) after being read.
                    (or None, the default, for no preprocessing).
    """
    viewL = modelnet40_filenames(subset, src_dir)
    def generator_func():
        repeat = 0
        while repeats is None or repeat < repeats:
            if shuffle:
                random.shuffle(viewL)
            for (i, view) in enumerate(viewL[:int(len(viewL)*frac)]):
                if verbose == 1 and i % 100 == 0:
                    print('Loading %s data: %.1f%%' % (subset, i*100.0/(len(viewL)*frac)))
                (cls, view) = view
                if verbose == 2:
                    print('Loading data point %d, cls = %d' % (i, cls))
                if class_array:
                    cls_array = numpy.zeros((1, nclasses), 'float32')
                    cls_array[0, cls] = 1.0
                    cls = cls_array
                if single:
                    filename = random.choice(view)
                    yield (read_image(filename, target_size, preprocess), cls)
                else:
                    yield ([read_image(view_elem, target_size, preprocess) for view_elem in view], cls)
            repeat += 1 
    return (generator_func(), len(viewL))

if __name__ == '__main__':
    
    (g, dataset_size) = modelnet40_generator('test')
    print('Loading first element from dataset')
    (x1, y1) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
    print(x1, y1)
    print('Loading second element from dataset')
    (x2, y2) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
    print(x2, y2)
    print('Done loading')
    print(x2.shape)
    
    """   
    model=applications.ResNet50(weights='imagenet', include_top=False)
    print('Model loaded.')
    print(model.output_shape)
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Flatten(batch_input_shape=model.output_shape[1:]))
    top_model.add(Dense(40, activation='softmax'))
    """
    """
    model = applications.ResNet50(weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    """
    
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50

    base_model=applications.ResNet50(weights='imagenet',input_shape=x2.shape[1:], include_top=False)
    print('Model loaded.')
    print(base_model.output_shape)
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
    train_generator= modelnet40_generator('train', src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    validation_generator=modelnet40_generator('test', src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None, shuffle=True,  verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input)
    whole_model.fit_generator(train_generator[0], samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=validation_generator[0],nb_val_samples=nb_validation_samples)
   
