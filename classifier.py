# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from keras import layers
from keras import models
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import KFold

from mobilenetv1_encoder import Encoder, load_img
from sql_models import db_handler, Sight, Descriptor
from algorithm_for_choosing import split_by_labels, unite_by_labels, sight_cluster, score_clustering
from utils import plot_history
import peewee
import csv
import json
from tqdm import tqdm
import argparse

import numpy as np


classes_count = 20


class Classifier:
	def __init__(self, cnt_classes, input_shape, loss, weights_file=None):
		self.model = models.Sequential()
		self.model.add(layers.Reshape((1, 1, input_shape[0]), input_shape=input_shape))
		self.model.add(layers.Dropout(rate=0.07))
		self.model.add(layers.Conv2D(cnt_classes, (1, 1), padding='same'))
		self.model.add(layers.Activation('relu'))
		self.model.add(layers.Dropout(rate=0.07))
		self.model.add(layers.Conv2D(cnt_classes, (1, 1), padding='same'))
		self.model.add(layers.Activation('softmax'))
		self.model.add(layers.Reshape((cnt_classes,)))

		self.cnt_classes = cnt_classes
		if weights_file:
			self.model.load_weights(weights_file)

		self.model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])

	def predict(self, desc):
		if len(desc.shape) == 1:
			desc = np.expand_dims(desc, 0)

		return self.model.predict(desc)

	def fit(self, x, y, kfold=10):
		y = to_categorical(y, self.cnt_classes)
		for train_index, test_index in KFold(kfold, True).split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]

			return self.model.fit(x_train, y_train, batch_size=100, epochs=150, validation_data=(x_test, y_test), shuffle=True)


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


def read_csv(fname, sights: dict, nas = False, count = None):
	csvfile = open(fname, newline='')
	reader = csv.reader(csvfile)

	if not count:
		count = sum([1 if (int(row[0]) in sights) else 0 for row in reader])
	x_train = np.zeros((count, 1024))
	y_train = np.zeros((count,))

	progress = tqdm(total=count)
	csvfile.seek(0)
	i = 0
	nas_index = max(sights.values()) + 1
	for row in reader:
		image_id = int(row[0])
		descr = row[1:]

		if nas:
			if (not (image_id in sights)) and i < count:
				x_train[i] = np.asarray(descr)
				y_train[i] = nas_index
				i += 1
				progress.update(1)
		else:
			if image_id in sights:
				x_train[i] = np.asarray(descr)
				y_train[i] = sights[image_id]
				i += 1
				progress.update(1)
		if i == count:
			break

	progress.close()
	return (x_train, y_train)


def areas_cluster(x, y, classes):
	areas = split_by_labels(x, y)
	extra_classes = []
	extra_areas = []

	for i, area in enumerate(areas):
		labels = sight_cluster(area, 'kmeans')

		count = np.max(labels)+1
		print(f'{count} clusters on {classes[i]}')
		score = np.inf

		counts = np.unique(labels, return_counts=True)[1]
		median = np.median(counts)
		min = np.min(counts)
		if count > 1:
			score = score_clustering(area, labels)
			print('Score:', score, 'median:', median, 'min:', min)

		if (median > 200) and (score < 200):
			t = split_by_labels(area, labels)
			extra_classes += [classes[i]] * (count-1)
			extra_areas += t[1:]
			areas[i] = t[0]

	areas += extra_areas
	x, y = unite_by_labels(areas)
	x = np.array(x)
	y = np.array(y)

	print('Added', len(extra_classes))
	return x, y, classes+extra_classes


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
		x, y = read_csv(args.fin, sights, False)
		x_nas, y_nas = read_csv(args.fin, sights, True, x.shape[0])
	else:
		x, y = read_sql()

	x, y, classes = areas_cluster(x, y, classes)
	x = np.append(x, x_nas, axis=0)
	y = np.append(y, y_nas, axis=0)
	classes.append(-1)

	model = Classifier(len(classes), x.shape[1:], 'categorical_crossentropy')
	history = model.fit(x, y)

	with open(args.hist_file, 'w') as f:
		f.write(json.dumps(history.history))
	with open(args.classes_file, 'w') as f:
		f.write(json.dumps(classes))
	model.model.save_weights(args.weights_file)
	plot_history(history.history)
