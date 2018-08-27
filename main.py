import math, shutil, os, time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel
import datetime
import os
import matplotlib.pyplot as plt

'''
Train/test code for iTracker.

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


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Gaze(object):
	"""docstring for ."""
	def __init__(self):
		print ("----init----")

		# Change there flags to control what happens.
		self.doLoad = False # Load checkpoint at the beginning
		self.doTest = False # Only run test, no training

		self.workers = 8
		self.epochs = 100
		self.batch_size = 10
		# torch.cuda.device_count()*100 # Change if out of cuda memory
		# batch_size = 10
		self.base_lr = 0.0001
		self.momentum = 0.9
		self.weight_decay = 1e-4
		self.print_freq = 10
		self.prec1 = 0
		self.best_prec1 = 1e20
		self.lr = self.base_lr


		self.train_loss_his, self.prec1_his, self.val_error_his = [], [], []

		now = datetime.datetime.now()
		self.date = now.strftime("%Y-%m-%d-%H-%M")

		self.CHECKPOINTS_PATH = 'my_model/' + self.date + "/"
		self.plot_ckpt = "plots/" + self.date + "/"
		if not os.path.exists(self.CHECKPOINTS_PATH):
			os.makedirs(self.CHECKPOINTS_PATH)
		if not os.path.exists(self.plot_ckpt):
			os.makedirs(self.plot_ckpt)
		print ("----finish init----")

	def main(self):
		print ("----main----")
		# global args, best_prec1, weight_decay, momentum

		model = ITrackerModel()
		model = torch.nn.DataParallel(model)
		model.cuda()
		imSize=(224,224)
		cudnn.benchmark = True

		epoch = 0
		if self.doLoad:
			saved = load_checkpoint()
			if saved:
				print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_prec1']))
				state = saved['state_dict']
				try:
					model.module.load_state_dict(state)
				except:
					model.load_state_dict(state)
				epoch = saved['epoch']
				self.best_prec1 = saved['best_prec1']
			else:
				print('Warning: Could not read checkpoint!');


		dataTrain = ITrackerData(split='train', imSize = imSize)
		dataVal = ITrackerData(split='test', imSize = imSize)
		# raise "debug"

		train_loader = torch.utils.data.DataLoader(
			dataTrain,
			batch_size=self.batch_size, shuffle=False,
			num_workers=self.workers, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			dataVal,
			batch_size=self.batch_size, shuffle=False,
			num_workers=self.workers, pin_memory=True)


		criterion = nn.MSELoss().cuda()

		optimizer = torch.optim.SGD(model.parameters(), self.lr,
									momentum=self.momentum,
									weight_decay=self.weight_decay)

		print ("================== batch_size ==================")
		print ("==================", self.batch_size, "==================")

		# Quick test
		if self.doTest:
			print ("validate: ")
			self.validate(val_loader, model, criterion, epoch)
			return

		# for epoch in range(0, epoch):
		#     adjust_learning_rate(optimizer, epoch)

		for epoch in range(epoch, self.epochs):
			print ("epoch: ", epoch)
			self.adjust_learning_rate(optimizer, epoch)

			# train for one epoch
			self.train(train_loader, model, criterion, optimizer, epoch, val_loader)

			# evaluate on validation set
			(prec1, val_error) = self.validate(val_loader, model, criterion, epoch)

			# remember best prec@1 and save checkpoint
			is_best = prec1 < self.best_prec1
			self.best_prec1 = min(prec1, self.best_prec1)
			self.save_checkpoint(is_best, epoch, "max",  prec1, val_error, {
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_prec1': self.best_prec1,
			})


	def train(self, train_loader, model, criterion, optimizer, epoch, val_loader):
		print ("----train----")

		# global count
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()

		# switch to train mode
		model.train()

		end = time.time()

		train_loss = []

		for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):

			# measure data loading time
			data_time.update(time.time() - end)
			imFace = imFace.cuda(async=True)
			imEyeL = imEyeL.cuda(async=True)
			imEyeR = imEyeR.cuda(async=True)
			faceGrid = faceGrid.cuda(async=True)
			gaze = gaze.cuda(async=True)

			imFace = torch.autograd.Variable(imFace)
			imEyeL = torch.autograd.Variable(imEyeL)
			imEyeR = torch.autograd.Variable(imEyeR)
			faceGrid = torch.autograd.Variable(faceGrid)
			gaze = torch.autograd.Variable(gaze)

			# compute output
			output = model(imFace, imEyeL, imEyeR, faceGrid)

			loss = criterion(output, gaze)

			losses.update(loss.data[0], imFace.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			train_loss.append(loss.data[0])


			if i % 10 == 0:
				train_loss_mean = np.mean(train_loss)
				print ("train_loss: ", train_loss_mean)
				(prec1, val_error) = self.validate(val_loader, model, criterion, epoch)

				self.save_checkpoint(False, epoch, i, prec1, val_error, {
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_prec1': None,
				})

				print('Epoch (train): [{0}][{1}/{2}]\t'
						  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
						   epoch, i, len(train_loader), batch_time=batch_time,
						   data_time=data_time, loss=losses))

				print ("prec1: ", prec1)
				print (type(prec1))

				self.train_loss_his.append(train_loss_mean)
				self.val_error_his.append(val_error)
				self.prec1_his.append(prec1)

				print (self.train_loss_his[:10])
				print (self.val_error_his[:10])
				print (self.prec1_his[:10])
				raise "debug"

				self.plot_loss(self.train_loss_his, self.val_error_his, self.prec1_his, save_file = self.plot_ckpt + "/cumul_loss_" + str(epoch) + "_" + str(i) + ".png")

	def validate(self, val_loader, model, criterion, epoch):
		print ("----validate----")

		# global count_test
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		lossesLin = AverageMeter()

		# switch to evaluate mode
		model.eval()
		end = time.time()

		val_loss = []

		for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			imFace = imFace.cuda(async=True)
			imEyeL = imEyeL.cuda(async=True)
			imEyeR = imEyeR.cuda(async=True)
			faceGrid = faceGrid.cuda(async=True)
			gaze = gaze.cuda(async=True)
	# TODO:
			imFace = torch.autograd.Variable(imFace, volatile = True)
			imEyeL = torch.autograd.Variable(imEyeL, volatile = True)
			imEyeR = torch.autograd.Variable(imEyeR, volatile = True)
			faceGrid = torch.autograd.Variable(faceGrid, volatile = True)
			gaze = torch.autograd.Variable(gaze, volatile = True)

			# compute output
			output = model(imFace, imEyeL, imEyeR, faceGrid)

			loss = criterion(output, gaze)

			lossLin = output - gaze
			lossLin = torch.mul(lossLin,lossLin)
			lossLin = torch.sum(lossLin,1)
			lossLin = torch.mean(torch.sqrt(lossLin))

			losses.update(loss.data[0], imFace.size(0))
			lossesLin.update(lossLin.data[0], imFace.size(0))

			# compute gradient and do SGD step
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			val_loss.append(loss.data[0])

			# if i % 10 == 0:
		print ("val_loss: ", np.mean(val_loss))
		print('Epoch (val): [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
					epoch, i, len(val_loader), batch_time=batch_time,
				   loss=losses,lossLin=lossesLin))
			# i += 1
		print ("val_loss: ", np.mean(val_loss))
		return (lossesLin.avg, np.mean(val_loss))



	def load_checkpoint(self, filename='checkpoint.pth.tar'):
		filename = os.path.join(self.CHECKPOINTS_PATH, filename)
		print(filename)
		if not os.path.isfile(filename):
			return None
		state = torch.load(filename)
		return state

	def save_checkpoint(self, is_best, epoch, iter, prec1, val_error, state = None, filename='checkpoint.pth.tar'):
		print ("----save checkpoint----")

		if not os.path.isdir(self.CHECKPOINTS_PATH):
			os.makedirs(self.CHECKPOINTS_PATH, 0o777)
		bestFilename = os.path.join(self.CHECKPOINTS_PATH, 'best_' + filename)
		filename = os.path.join(self.CHECKPOINTS_PATH, str(epoch) + "_" + str(prec1) + "_" + str(val_error) + str(iter) + "_" + filename)
		torch.save(state, filename)
		if is_best:
			shutil.copyfile(filename, bestFilename)


	def plot_loss(self, train_loss_his, val_error_his, prec1_his, start=0, per=1, save_file='loss.png'):
		print ("----plot loss----")

		idx = np.arange(start, len(train_loss_his), per)
		fig, ax1 = plt.subplots()
		lns1 = ax1.plot(idx, train_loss_his[idx], 'b-', alpha=1.0, label='train loss')
		ax1.set_xlabel('epochs')
		# Make the y-axis label, ticks and tick labels match the line color.
		ax1.set_ylabel('loss', color='b')
		ax1.tick_params('y', colors='b')

		ax2 = ax1.twinx()
		lns2 = ax2.plot(idx, val_error_his[idx], 'r-', alpha=1.0, label='val error')
		lns3 = ax2.plot(idx, prec1_his[idx], 'g-', alpha=1.0, label='prec1')
		ax2.set_ylabel('error', color='r')
		ax2.tick_params('y', colors='r')

		# added these three lines
		lns = lns1 + lns2 + lns3
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc=0)

		fig.tight_layout()
		plt.savefig(save_file)
		# plt.show()

	def adjust_learning_rate(self, optimizer, epoch):
		print ("----adjust learning rate----")

		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		self.lr = self.base_lr * (0.1 ** (epoch // 30))
		for param_group in optimizer.state_dict()['param_groups']:
			param_group['lr'] = self.lr


if __name__ == "__main__":
	Gaze().main()
	print('DONE')
