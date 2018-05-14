import cv2
import numpy as np
import os

path = "./ML/Fnt32"

for folder in os.listdir(path):
	print(folder)
	for img_path in os.listdir(path+"/"+folder):

		image = cv2.imread(path+"/"+folder+"/"+img_path)
		height, width, channels = image.shape
		#if(height>=32 and width >=32):

		resized_img = cv2.resize(image,(32,32))
		cv2.imwrite("./test/"+folder+"/"+img_path,resized_img)
