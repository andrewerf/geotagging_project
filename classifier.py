# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from keras import layers
from keras import models
import argparse
from keras.preprocessing import image
from keras.utils import to_categorical
from geotagging_project.mobilenetv1_encoder import Encoder
from geotagging_project.sql_models import db_handler, Sight, Descriptor
import csv
from tqdm import tqdm

import numpy as np

classes = 70


class Classifier:
	def __init__(self, cnt_classes, input_shape):
		self.model = models.Sequential()
		self.model.add(layers.Reshape((1, 1, input_shape[0]), input_shape=input_shape))
		self.model.add(layers.Dropout(rate=1e-3))
		self.model.add(layers.Conv2D(cnt_classes, (1, 1), padding='same'))
		self.model.add(layers.Activation('softmax'))
		self.model.add(layers.Reshape((cnt_classes,)))
		self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

	def predict(self, desc):
		return self.model.predict(desc)

	def fit(self, x, y):
		x_test = x[:1000]
		x_train = x[1000:]
		y_test = y[:1000]
		y_train = y[1000:]

		y_train = to_categorical(y_train)
		y_test = to_categorical(y_test)

		self.model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), shuffle=True)


def read_csv(fname, sights: dict):
	csvfile = open(fname, newline='')
	reader = csv.reader(csvfile)

	count = sum([1 if (int(row[0]) in sights) else 0 for row in reader])
	x_train = np.zeros((count, 1024))
	y_train = np.zeros((count,))

	progress = tqdm(total=count)
	csvfile.seek(0)
	i = 0
	for row in reader:
		image_id = int(row[0])
		if not (image_id in sights):
			continue

		descr = row[1:]

		x_train[i] = np.asarray(descr)
		y_train[i] = sights[image_id]
		i += 1

		progress.update(1)

	progress.close()
	return (x_train, y_train)


def read_sql():
	return (0, 0)


def load_img(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	img = image.img_to_array(img)
	return img


def test(img_path):
	img = load_img(img_path)
	en = Encoder()
	desc = en.get_descriptor(img, False)

	model = Classifier(70, desc.shape)

	desc = np.expand_dims(desc, 0)
	pred = model.predict(desc)

	print(np.argmax(pred[0]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, dest='fin')
	args = parser.parse_args()

	sights = {}
	query = Descriptor.select(Descriptor.image_id, Descriptor.sight_id).tuples()
	for row in tqdm(query):
		if row[1] < classes:
			sights[int(row[0])] = row[1]

	if args.fin:
		x, y = read_csv(args.fin, sights)
	else:
		x, y = read_sql()

	model = Classifier(classes, (1024,))
	model.fit(x, y)
