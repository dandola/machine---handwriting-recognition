import matplotlib
import h5py
import time
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten,Dropout
import load_data
from keras.models import model_from_json
import slicing_data
from keras.optimizers import SGD
matplotlib.use('Agg')
def main():
	num_classes=62
	batch_size=128
	epochs= 5
	img_rows,img_cols=32,32
	training_data_directory="./ML/Fnt32"

	x_train,x_test,y_train,y_test= load_data.load_data(training_data_directory)
	print("len x_train: ", len(x_train))
	print("len x_test: ", len(x_test))

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	start_time=time.time()
	model=Sequential()
	#layer Conv2D_1
	model.add(Conv2D(32,3,padding="same",activation='relu',input_shape=(img_cols,img_cols,3)))
	model.add(Conv2D(32,3,activation='relu'))
	#layer MaxPololing2D_1
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	#layer Conv2D_2
	model.add(Conv2D(64,3,padding="same",activation='relu'))
	model.add(Conv2D(64,3,activation='relu'))
	#layer MaxPololing2D_2
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Conv2D(64,3,padding="same",activation='relu'))
	#layer MaxPololing2D_2
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Flatten())
	#layer Dense 1
	model.add(Dense(units=1024,activation='relu'))
	model.add(Dropout(0.5))
	#layer Dense 2
	model.add(Dense(62, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])
	print(model.summary())
	#training
	callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0,)]
	history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, shuffle=True,callbacks=callbacks, validation_data=(x_test,y_test))
	#evaluation
	end_time= time.time()
	total_time = (end_time - start_time)
	print("Time to training: ", total_time, " seconds")
	model.evaluate(x_test,y_test, verbose=0)
	# model_json=model.to_json()
	# with open("model.json","w") as f:
		# f.write(model_json)
	# serialize weights to HDF5
	# model.save_weights("model.h5")
	# print("Saved model to disk")
	
	print(history.history.keys)
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['acc', 'val_acc'], loc='upper left')
	plt.savefig("model_accuracy")
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['loss', 'val_loss'], loc='upper left')
	plt.savefig("model_loss")
	plt.show()
main()