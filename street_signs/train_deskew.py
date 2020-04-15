import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

from street_signs.unet import create_unet
from street_signs.fcnn_loader import fcnn_loader
from utils import visualize

import imgaug.augmenters as aug
from keras import optimizers, losses


def get_preprocessing(shape):

	def wrapper(img, mask):
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

		affine_scale = 0.7
		seq = aug.Sequential([
			aug.Affine(scale=affine_scale, cval=255),
			aug.Rotate((-10, 10), cval=255),
			aug.PerspectiveTransform((0.05, 0.1), cval=255)
		])

		img = seq.augment_image(img)
		mask = aug.Affine(scale=affine_scale, cval=255).augment_image(mask)

		mask = cv2.resize(mask, shape)
		img = cv2.resize(img, shape)

		# visualize(img=img, mask=mask)

		img = np.expand_dims(img, 2)
		mask = np.expand_dims(mask, 2)

		return img / 255, mask / 255

	return wrapper


def main():

	batch_size = 5
	optimizer = optimizers.SGD(momentum=0.9)
	loss = losses.binary_crossentropy
	metrics = []
	epochs = 10
	target_shape = (112, 80)

	loader = fcnn_loader('/media/andrew/Data/Temp/nn/mnist-words/dataset/v011_words_small', '/media/andrew/Data/Temp/nn/mnist-words/dataset/v011_words_small', batch_size, get_preprocessing(target_shape))
	train_loader, test_loader = loader.split_train_test(0.1)


	model = create_unet((target_shape[1], target_shape[0], 1), 1)
	model.compile(optimizer, loss, metrics)
	model.fit_generator(train_loader, len(train_loader),
						validation_data=test_loader, validation_steps=len(test_loader),
						epochs=epochs)


if __name__ == '__main__':
	main()