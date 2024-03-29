import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import pickle
import cv2

'''
Data loader for the iTracker.
Modify to fit your data format.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018.

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

DATASET_PATH = "../../Eye-Tracking-for-Everyone-master/Eye-Tracking-for-Everyone-master/GazeCapture/"
MEAN_PATH = './'
META_PATH = './metadata.mat'

def loadMetadata(filename, silent = False):
	try:
		# http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
		if not silent:
			print('\tReading metadata from %s...' % filename)
		metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
	except:
		print('\tFailed to read the meta file "%s"!' % filename)
		return None
	return metadata

class SubtractMean(object):
	"""Normalize an tensor image with mean.
	"""

	def __init__(self, meanImg):
		self.meanImg = transforms.ToTensor()(meanImg)

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		return tensor.sub(self.meanImg)


class ITrackerData(data.Dataset):
	def __init__(self, split = 'train', imSize=(224,224), gridSize=(25, 25)):
		self.good_counter = 0
		self.bad_counter = 0

		self.imSize = imSize
		self.gridSize = gridSize

		print('Loading iTracker dataset...')
		self.metadata = loadMetadata(META_PATH)
		self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
		self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
		self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']


		self.transformFace = transforms.Compose([
			transforms.Scale(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.faceMean),
		])
		self.transformEyeL = transforms.Compose([
			transforms.Scale(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.eyeLeftMean),
		])
		self.transformEyeR = transforms.Compose([
			transforms.Scale(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.eyeRightMean),
		])


		if split == 'test':
			# mask = self.metadata['labelTest']
			self.indices = pickle.load( open( "test_Indices.p", "rb" ) )
			# [:1000]

			# self.indices = pickle.load( open( "train_Indices.p", "rb" ) )
			# [1000:1100]

		elif split == 'val':
			# mask = self.metadata['labelVal']
			pass
		else:
			# mask = self.metadata['labelTrain']
			self.indices = pickle.load( open( "train_Indices.p", "rb" ) )
			# [:1000]

		# self.indices = np.argwhere(mask)[:,0]


		print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))


		# print (len(self.indices))

		# try:
		# 	self.indices = pickle.load( open( split + "_indices_test.p", "rb" ) )
		# except Exception as e:

		# self.check_indices(split)

	def convert_image_cv(self, img):
		open_cv_image = np.array(img)
		# Convert RGB to BGR
		open_cv_image = open_cv_image[:, :, ::-1].copy()
		return open_cv_image

	def loadImage(self, path):
		try:
			im = Image.open(path).convert('RGB')
			return im
		except Exception as e:
			# print (e)
		# except OSError:
		# 	raise RuntimeError('Could not read image: ' + path)
			pass
			#im = Image.new("RGB", self.imSize, "white")



	def makeGrid(self, params):
		gridLen = self.gridSize[0] * self.gridSize[1]
		grid = np.zeros([gridLen,], np.float32)

		indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
		indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
		condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
		condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
		cond = np.logical_and(condX, condY)

		grid[cond] = 1
		return grid

	def check_indices(self, split):
		print ("checking indices")
		tmp_indices = []
		for i in range(len(self.indices)):
			if i % 1000 == 0:
				print ("i: ", i)
				print ("len(tmp_indices): ", len(tmp_indices))

			index = self.indices[i]
			imFacePath = os.path.join(DATASET_PATH, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
			imEyeLPath = os.path.join(DATASET_PATH, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
			imEyeRPath = os.path.join(DATASET_PATH, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

			try:
				imFace = self.loadImage(imFacePath)
				imEyeL = self.loadImage(imEyeLPath)
				imEyeR = self.loadImage(imEyeRPath)

				imFace = self.transformFace(imFace)
				imEyeL = self.transformEyeL(imEyeL)
				imEyeR = self.transformEyeR(imEyeR)

				gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

				faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

				# to tensor
				row = torch.LongTensor([int(index)])
				faceGrid = torch.FloatTensor(faceGrid)
				gaze = torch.FloatTensor(gaze)
				# print ("working: ", index)
				tmp_indices.append(index)
			except Exception as e:
				pass
				# print (e)
				# print ("failing: ", index)

			# if i > 2000:
			# 	break

		self.indices = tmp_indices
		pickle.dump(self.indices, open(split + "_indices_test.p", "wb"))
		print ("finish checking")

	def __getitem__(self, index):
		index = self.indices[index]

		imFacePath = os.path.join(DATASET_PATH, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
		imEyeLPath = os.path.join(DATASET_PATH, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
		imEyeRPath = os.path.join(DATASET_PATH, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

		imFace = self.loadImage(imFacePath)
		imEyeL = self.loadImage(imEyeLPath)
		imEyeR = self.loadImage(imEyeRPath)


		gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

		faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

		# y_x, y_y = gaze
		# increase = 3
		# y_x, y_y = - int(y_x * increase), int(y_y * increase)
		# h, w = imFace.size

		# imFace_cv = self.convert_image_cv(imFace)
		# imEyeL_cv = self.convert_image_cv(imEyeL)
		# imEyeR_cv = self.convert_image_cv(imEyeR)
		#
		# cx, cy = w/2.0, h/2.0
		# cv2.circle(imFace_cv,(int(cx), int(cy)), 5, (0,0,255), -1)
		# cv2.line(imFace_cv, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)
		#
		# cv2.imwrite("images/" + str(index) + "_face.png", imFace_cv)
		# cv2.imwrite("images/" + str(index) + "_right.png", imEyeR_cv)
		# cv2.imwrite("images/" + str(index) + "_left.png", imEyeL_cv)


		imFace = self.transformFace(imFace)
		imEyeL = self.transformEyeL(imEyeL)
		imEyeR = self.transformEyeR(imEyeR)

		# to tensor
		row = torch.LongTensor([int(index)])
		faceGrid = torch.FloatTensor(faceGrid)
		gaze = torch.FloatTensor(gaze)

		return row, imFace, imEyeL, imEyeR, faceGrid, gaze

	def __len__(self):
		return len(self.indices)



#     def __getitem__(self, index):
# # ////////////////////////////////////////
# # TODO:figure out what is index about
#
#         print ("vvvvvvvvvv")
#         print ("index: ", index)
#         if_load = False
#
#         index = self.indices[index]
#         print ("^^^^^^^^^^^")
#
#         raise "debug"
#         # print ("self.indices[index]: ", self.indices[index])
#
#         while not if_load:
#             try:
#                 imFacePath = os.path.join(DATASET_PATH, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
#                 imEyeLPath = os.path.join(DATASET_PATH, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
#                 imEyeRPath = os.path.join(DATASET_PATH, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
#
#                 print ("imFacePath: ", imFacePath)
#
#                 imFace = self.loadImage(imFacePath)
#                 imEyeL = self.loadImage(imEyeLPath)
#                 imEyeR = self.loadImage(imEyeRPath)
#
#                 self.good_counter += 1
#                 print ("self.good_counter: ", self.good_counter)
#                 print ("load image !!!!!!!!!!!!!!")
#                 imFace = self.transformFace(imFace)
#                 imEyeL = self.transformEyeL(imEyeL)
#                 imEyeR = self.transformEyeR(imEyeR)
#
#                 gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
#
#                 faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])
#
#                 # to tensor
#                 row = torch.LongTensor([int(index)])
#                 faceGrid = torch.FloatTensor(faceGrid)
#                 gaze = torch.FloatTensor(gaze)
#                 if_load = True
#
#             except Exception as e:
#                 self.bad_counter += 1
#                 # print ("self.bad_counter: ", self.bad_counter)
#                 # print (e)
#                 index += 1
#                 # return None, None, None, None, None, None
#
#         return row, imFace, imEyeL, imEyeR, faceGrid, gaze
#         # index,
