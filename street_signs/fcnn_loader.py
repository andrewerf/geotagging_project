import os
import cv2
import numpy as np
from keras.utils import Sequence


class fcnn_loader(Sequence):
	def __init__(self, imgs_dir, masks_dir, batch_size, preprocessing=None, augmentation=None):
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir
		self.batch_size = batch_size
		self.preprocessing = preprocessing
		self.augmentation = augmentation

		self.x_list = list(map(lambda x: os.path.join(imgs_dir, x), sorted(os.listdir(imgs_dir))))
		self.y_list = list(map(lambda x: os.path.join(masks_dir, x), sorted(os.listdir(masks_dir))))

		t1 = len(self.x_list)
		t2 = len(self.y_list)
		if t1 != t2:
			raise ValueError()
		else:
			self.count = t1

	def split_train_test(self, p):
		test_count = int(p * self.count)

		test_indices = np.random.choice(self.count, test_count)
		self.x_list = np.asarray(self.x_list)
		self.y_list = np.asarray(self.y_list)

		test_x_list, test_y_list = self.x_list[test_indices], self.y_list[test_indices]
		train_x_list, train_y_list = np.delete(self.x_list, test_indices, 0), np.delete(self.y_list, test_indices, 0)

		train_loader = fcnn_loader(self.imgs_dir, self.masks_dir, self.batch_size, self.preprocessing, self.augmentation)
		test_loader = fcnn_loader(self.imgs_dir, self.masks_dir, self.batch_size, self.preprocessing, self.augmentation)

		train_loader.x_list, train_loader.y_list = map(list, [train_x_list, train_y_list])
		train_loader.count = len(train_x_list)
		test_loader.x_list, test_loader.y_list = map(list, [test_x_list, test_y_list])
		test_loader.count = len(test_x_list)

		return train_loader, test_loader

	def load_pair(self, i):
		x = cv2.imread(self.x_list[i])
		y = cv2.imread(self.y_list[i])

		if self.preprocessing is not None:
			x, y = self.preprocessing(x, y)

		return x,y

	def __getitem__(self, i):
		start = i*self.batch_size
		stop = (i+1)*self.batch_size

		x_batch = np.zeros((self.batch_size,) + (self.load_pair(0)[0].shape))
		y_batch = np.zeros((self.batch_size,) + (self.load_pair(0)[1].shape))
		for j in range(start, stop):
			x, y = self.load_pair(j)
			x_batch[j-start] = x
			y_batch[j-start] = y

		if self.augmentation is not None:
			x_batch, y_batch = self.augmentation(x_batch, y_batch)

		return x_batch, y_batch

	def __len__(self):
		return self.count // self.batch_size