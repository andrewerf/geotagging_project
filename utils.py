import os
from math import cos, fabs
import matplotlib.pyplot as plt

from sql_models import Descriptor, Sight


imgs_path = '/media/andrew/HDD_Data/Temp/nn/Images/'


def show_image(img_path):
	img = plt.imread(img_path)
	plt.imshow(img)
	plt.show()


def show_images(image_id, imgs_path):
	name = image_id + '.jpg'
	for root, dirs, files in os.walk(imgs_path):
		if name in files:
			img = plt.imread(os.path.join(imgs_path, root, name))
			plt.imshow(img)
			plt.show()


def look_sight(sight_id):
	descr_ids = Descriptor.select(Descriptor.id).join(Sight).where(Sight.id == sight_id)

	for id in descr_ids:
		image_id = Descriptor.select(Descriptor.image_id).where(Descriptor.id == id)[0].image_id
		show_images(image_id, imgs_path)


def plot_history(history):
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


def convert_accuracy(pos, acc_m):
	acc_m = acc_m/100

	lat_deg = acc_m / 360.0
	long_deg = cos(pos[0]) * acc_m / 360.0

	return (lat_deg, long_deg)

class Area:
	def __init__(self, pos, accuracy_m = 50):
		self.pos = pos
		self.delta = list(map(fabs, convert_accuracy(pos, accuracy_m)))

	def __contains__(self, pos):
		ok = True

		for i in [0,1]:
			ok = ok and (self.pos[i] - self.delta[i] <= pos[i] <= self.pos[i] + self.delta[i])

		return ok

	def __add__(self, other):
		pass

	def __str__(self):
		return str(self.pos) + ',' + str(self.delta)
