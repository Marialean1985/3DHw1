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
def rotateImage(im,row_axis, col_axis, channel_axis,fill_mode, cval,RangeRotation=True):
	if(RangeRotation==True):
		im=image.random_rotation(im,45, row_axis, col_axis, channel_axis,fill_mode, cval)
	else:
		angle= random.choice([-45,0,+45])
#		print("angle is :",angle)
		theta = np.pi / 180 * angle
		rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
		h, w = im.shape[row_axis], im.shape[col_axis]
		transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
		im = image.apply_transform(im, transform_matrix, channel_axis, fill_mode, cval)
	return im

def batchGenerator( SingleGenerator, batchSize, dataSetSize,single=True,augment=True,RangeRotation=True):
	def nestedGen():
		while True:
			for i in range(batchSize):
				(a,b)=SingleGenerator.__next__() 
#				print("len a is",len(a))
#				print(a[0])
				row_axis=0
				col_axis=1
				channel_axis=2
				fill_mode='nearest'
				cval=0.
				if(augment==True and single==True):
					a[0]=rotateImage(a[0],row_axis, col_axis, channel_axis,fill_mode, cval,RangeRotation)
				if(augment==True and single==False):
					for j in range(len(a)):
					#random horizontal flip along columns
						a[j][0]=rotateImage(a[j][0],row_axis, col_axis, channel_axis,fill_mode, cval,RangeRotation)
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
