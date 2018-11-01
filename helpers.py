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

	I = np.asarray(images, dtype=np.float32)
	meanImage = np.mean(I, axis=0)
	meanStd = np.std(I, axis=0)
	I = (I - meanImage) / meanStd
	return I

def getAllMasks(folderPath):
	if folderPath[len(folderPath) - 1] == '/':
		folderPath = folderPath[0:-1]

	listOfFiles = os.listdir(folderPath)
	images = []

	for file in listOfFiles:
		filePath = folderPath + '/' + file
		if os.path.isfile(filePath):
			images.append(cv2.imread(filePath))
			# print filePath
			# print cv2.imread(filePath).shape
			# pdb.set_trace()

	return np.asarray(images, dtype=object)

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
	indices = np.arange(0, X.shape[0])

	train_end = int(X.shape[0] * 0.8)
	return X[0:train_end], Y1[0:train_end], Y2[0:train_end], indices[0:train_end],\
			X[train_end:], Y1[train_end:], Y2[train_end:], indices[train_end:]

def getMasks():
	carMasks = getAllMasks('P&C dataset/mask/car')
	peopleMasks = getAllMasks('P&C dataset/mask/people')

	return carMasks, peopleMasks