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

data_predict_direction="./ML/Fnt32"
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

images= load_data_predict.load_data(data_predict_direction)
X_input= random.sample(images,k)
for i in range(len(X_input)):
	io.imshow(X_input[i])
	io.show()
	image = np.reshape(X_input[i],[1,28,28,3])
	prediction = model.predict(image)
	print(np.argmax(prediction[0]))
	print(max(prediction[0]))

# print(np.argmax(prediction))
# for i in range(len(X_input)):
# 	plt.subplot(1, 4, i+1)
# 	plt.axis('off')
# 	plt.imshow(X_input[i], cmap="gray")
# 	plt.subplots_adjust(wspace=0.5)
# plt.show()



