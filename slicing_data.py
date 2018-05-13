import os
import os.path
import skimage
import numpy as np
import skimage.data
from skimage import transform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage import color
import tensorflow as tf


def reshape(images):
	len_images=len(images)
	images_28=[transform.resize(image,(28,28)) for image in images]
	return images_28


training_data_directory="/home/danbka/Music/Machine/btl/ML/Fnt32"
def load_data(data_directory):
	x_train,y_train=[],[]
	x_val,y_val=[],[]

	directories=[d for d in os.listdir(data_directory)
					if os.path.isdir(os.path.join(data_directory,d))]
	labels=[]
	images=[]
	for d in directories:
		label_directory=os.path.join(data_directory,d)
		file_names=[os.path.join(label_directory,f)
					for f in os.listdir(label_directory)]
		i=0
		for f in file_names:
			if i< 200:
				x_val.append(color.gray2rgb(skimage.data.imread(f)))
				y_val.append(np.int32(d[-3:])-1)
			else:
				x_train.append(color.gray2rgb(skimage.data.imread(f)))
				y_train.append(np.int32(d[-3:])-1)
			i+=1

	x_train = reshape(x_train)
	x_val = reshape(x_val)
	#convert to numpy array
	x_train=np.array(x_train,dtype=np.float32)
	x_val=np.array(x_val,dtype=np.float32)
	#convert to numpy array
	y_train=np.array(y_train,dtype=np.int32)
	y_val=np.array(y_val,dtype=np.int32)

	return x_train,y_train,x_val,y_val




# x_train,y_train,x_val,y_val = load_data(training_data_directory)
# print("load xong")

# print x_train.shape
# print y_train.shape
# print x_val.shape
# print y_val.shape

