import cv2
import numpy as np
import os

path = "./Ml/Fnt32"

for folder in os.listdir(path):
	print(folder)
	for img_path in os.listdir(path+"/"+folder):
		image = cv2.imread(path+"/"+folder+"/"+img_path)
		print(image)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite("./test/"+folder+"/"+img_path,gray_image)
