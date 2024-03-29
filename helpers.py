import sys, os
import numpy as np
import pdb, cv2

def getAllImages(folderPath):
	if folderPath[len(folderPath) - 1] == '/':
		folderPath = folderPath[0:-1]

	images = []
	for i in range(2000):
		imagePath = folderPath + '/{0:06d}.jpg'.format(i)
		images.append(cv2.imread(imagePath))

	I = np.asarray(images, dtype=np.float32)
	meanImage = np.mean(I, axis=0)
	meanStd = np.std(I, axis=0)
	I = (I - meanImage) / meanStd
	return I

def getAllMasks(folderPath):
	if folderPath[len(folderPath) - 1] == '/':
		folderPath = folderPath[0:-1]

	images = []
	for i in range(2000):
		imagePath = folderPath + '/{0:06d}.png'.format(i)
		im = cv2.imread(imagePath)
		resized_im = cv2.resize(im, (27, 27))
		images.append(resized_im[:, :, 0][:, :, None])

	# return np.asarray(images, dtype=object)
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

def getMasks():
	carMasks = getAllMasks('P&C dataset/mask/car')
	peopleMasks = getAllMasks('P&C dataset/mask/people')

	return carMasks, peopleMasks

def getXandY():
	X = getAllImages('P&C dataset/img')
	Y1 = getBbox('label_car.txt')
	Y2 = getBbox('label_people.txt')
	indices = np.arange(0, X.shape[0])

	carMasks, peopleMasks = getMasks()

	train_end = int(X.shape[0] * 0.8)
	return X[0:train_end], Y1[0:train_end], Y2[0:train_end], carMasks[0:train_end], peopleMasks[0:train_end],\
			X[train_end:], Y1[train_end:], Y2[train_end:], carMasks[train_end:], peopleMasks[train_end:]
