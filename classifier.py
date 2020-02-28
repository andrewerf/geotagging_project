# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from keras import layers
from keras import models
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import KFold

from mobilenetv1_encoder import Encoder, load_img
from sql_models import db_handler, Sight, Descriptor
import peewee
import csv
import json
from tqdm import tqdm
import argparse

import numpy as np
from matplotlib import pyplot as plt


classes_count = 20


class Classifier:
	def __init__(self, cnt_classes, input_shape, weights_file=None):
		self.model = models.Sequential()
		self.model.add(layers.Reshape((1, 1, input_shape[0]), input_shape=input_shape))
		self.model.add(layers.Dropout(rate=0.05))
		self.model.add(layers.Conv2D(cnt_classes, (1, 1), padding='same'))
		self.model.add(layers.Activation('relu'))
		self.model.add(layers.Dropout(rate=0.05))
		self.model.add(layers.Conv2D(cnt_classes, (1, 1), padding='same'))
		self.model.add(layers.Activation('softmax'))
		self.model.add(layers.Reshape((cnt_classes,)))

		if weights_file:
			self.model.load_weights(weights_file)

		self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

	def predict(self, desc):
		if len(desc.shape) == 1:
			desc = np.expand_dims(desc, 0)

		return self.model.predict(desc)

	def fit(self, x, y):
		y = to_categorical(y, classes_count)
		for train_index, test_index in KFold(15, True).split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]

			return self.model.fit(x_train, y_train, batch_size=100, epochs=200, validation_data=(x_test, y_test), shuffle=True)

	def plot(self, history):
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.show()


def get_ktop_areas(k):
	ret = []
	ndescrs = peewee.fn.count(Descriptor.id)

	query = Sight.select(Sight.id, ndescrs.alias('count')).join(Descriptor, join_type=peewee.JOIN.LEFT_OUTER).group_by(Sight).order_by(ndescrs.desc())

	i = 0
	for row in query:
		if i == k:
			break
		ret.append(row.id)
		i += 1
	return ret


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


def test(img_path):
	img = load_img(img_path)
	en = Encoder()
	desc = en.get_descriptor(img, False)

	model = Classifier(classes_count, desc.shape)

	desc = np.expand_dims(desc, 0)
	pred = model.predict(desc)

	print(np.argmax(pred[0]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, dest='fin')
	parser.add_argument('--history', type=str, dest='hist_file')
	parser.add_argument('--weights', type=str, dest='weights_file')
	parser.add_argument('--classes', type=str, dest='classes_file')
	args = parser.parse_args()

	sights = {}
	classes = get_ktop_areas(classes_count)
	query = Descriptor.select(Descriptor.image_id, Descriptor.sight_id).tuples()
	for row in tqdm(query):
		if row[1] in classes:
			sights[int(row[0])] = classes.index(row[1])

	if args.fin:
		x, y = read_csv(args.fin, sights)
	else:
		x, y = read_sql()

	model = Classifier(classes_count, x.shape[1:])
	history = model.fit(x, y)

	with open(args.hist_file, 'w') as f:
		f.write(json.dumps(history.history))
	with open(args.classes_file, 'w') as f:
		f.write(json.dumps(classes))
	model.model.save_weights(args.weights_file)
	model.plot(history)
