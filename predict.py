from keras.models import load_model
import numpy as np
import matplotlib
import h5py
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,Dropout
import load_data
from keras.models import model_from_json
import slicing_data
from keras.optimizers import SGD
import load_data_predict
import random
from skimage import io

data_predict_direction="./ML/Predict"
k=10

num_classes=62
img_rows,img_cols=28,28

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
print(model.summary())
images= load_data_predict.load_data(data_predict_direction)
X_input= random.sample(images,k)
for i in range(len(X_input)):
	io.imshow(X_input[i])
	io.show()
	image = np.reshape(X_input[i],[1,28,28,3])
	prediction = model.predict(image)
	position= np.argmax(prediction[0])
	print("thuoc class: ",np.argmax(prediction[0]) + 1)
	if position < 10:
		c = chr(position+48)
	elif position >=10 and position < 36:
		c= chr(position + 65 - 10)
	else:
		c=chr(position + 97 - 36)
	print"predict - anh tren la ky tu: ",c ," voi xac suat la: ",round(max(prediction[0])*100,2),"%"




