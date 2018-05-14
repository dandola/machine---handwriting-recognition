import os
import os.path
import skimage
import random
import numpy as np
import skimage.data
from skimage import transform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage import color
import tensorflow as tf
from sklearn.model_selection import train_test_split


# training_data_directory="/home/danbka/Music/Machine/btl/ML/Fnt32"

def load_data(data_directory):
	directories=[d for d in os.listdir(data_directory)
					if os.path.isdir(os.path.join(data_directory,d))]
	labels=[]
	images=[]
	for d in directories:
		label_directory=os.path.join(data_directory,d)
		file_names=[os.path.join(label_directory,f)
					for f in os.listdir(label_directory)]
		for f in file_names:
			images.append(color.gray2rgb(skimage.data.imread(f)))
			labels.append(np.int32(d[-3:])-1)
	len_images=len(images)
	images_28=[transform.resize(image,(28,28)) for image in images]
	images=np.array(images_28,dtype=np.float32)
	labels=np.array(labels,dtype=np.int32)
	X_train, X_test, y_train, y_test = train_test_split(images,labels, test_size=0.2)	
	return X_train, X_test, y_train, y_test

def delete_all(data_directory):
	directories=[d for d in os.listdir(data_directory)
					if os.path.isdir(os.path.join(data_directory,d))]
	for d in directories:
		label_directory=os.path.join(data_directory,d)
		file_names=[os.path.join(label_directory,f)
					for f in os.listdir(label_directory)]
		for f in file_names:
			os.remove(f)
	return True


def split_predict(data_directory):
	path="/home/danbka/Music/Machine/btl/ML/Predict"
	directories=[d for d in os.listdir(data_directory)
					if os.path.isdir(os.path.join(data_directory,d))]
	for d in directories:
		label_directory=os.path.join(data_directory,d)
		file_names=[f for f in os.listdir(label_directory)]
		links = random.sample(file_names, 10)
		dir= os.path.join(path,d)
		for link in links:
			source=os.path.join(label_directory,link)
			destination=os.path.join(dir,link)
			os.rename(source,destination)
	return True

