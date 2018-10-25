import sys, os
import numpy as np
import pdb, cv2

def getAllImages(folderPath):
	if folderPath[len(folderPath) - 1] == '/':
		folderPath = folderPath[0:-1]

	listOfFiles = os.listdir(folderPath)
	images = []

	for file in listOfFiles:
		filePath = folderPath + '/' + file
		if os.path.isfile(filePath):
			images.append(cv2.imread(filePath))

	return np.asarray(images, dtype=np.float32)

def getBbox(filePath):
	f = open(filePath, 'r')
	lines = f.readlines()
	bboxes = []
	for line in lines:
		if line[-1] == '\n':
			line = line[0:-1]
		bboxes.append(line.split(','))
	return np.asarray(bboxes, dtype=np.float32)

def getXandY():
	X = getAllImages('P&C dataset/img')
	Y1 = getBbox('label_car.txt')
	Y2 = getBbox('label_people.txt')

	train_end = int(X.shape[0] * 0.8)
	return X[0:train_end], Y1[0:train_end], Y2[0:train_end],\
			X[train_end:], Y1[train_end:], Y2[train_end:]
