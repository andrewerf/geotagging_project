import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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



class Classifier:
	def __init__(self, cnt_classes, input_shape, weights_file=None):
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

		if cnt_classes == 2:
			loss = 'binary_crossentropy'
		else:
			loss = 'categorical_crossentropy'
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


class OneVsAllCascade:
	def __init__(self, cnt_classes, input_shape, weights_path=None):
		weights_files = [None for i in range(cnt_classes)]
		if weights_path:
			for k, fname in enumerate(os.listdir(weights_path)):
				t = int(os.path.splitext(fname)[0])
				weights_files[t] = os.path.join(weights_path, fname)

		self.classifiers = []
		for k in range(cnt_classes):
			c = Classifier(2, input_shape, weights_files[k])
			self.classifiers.append(c)

	def predict(self, desc, min_confidence):
		if len(desc.shape) == 1:
			desc = np.expand_dims(desc, 0)
		preds = np.asarray([classifier.predict(desc)[0] for classifier in self.classifiers])
		if np.max(preds[:,0]) < min_confidence:
			return len(self.classifiers)+1
		else:
			return np.argmax(preds[:,0])

	def fit(self, x, y):
		data = split_by_labels(x, y)
		nas = np.asarray(data[-1])
		data.pop()

		histories = []
		for k in range(len(data)):
			c_x = np.asarray(data[k])
			c_y = np.asarray([0 for i in range(c_x.shape[0])])

			indices = np.random.choice(nas.shape[0], c_x.shape[0], replace=False)

			not_x = nas[indices]
			not_y = np.asarray([1 for i in range(not_x.shape[0])])

			histories.append(self.classifiers[k].fit(np.append(c_x, not_x, axis=0), np.append(c_y, not_y)))

		return histories

	def save_weights(self, path):
		if not os.path.exists(path):
			os.mkdir(path)

		for k, classifier in enumerate(self.classifiers):
			classifier.model.save_weights(os.path.join(path, str(k) + '.h5'))


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
			if (not (image_id in sights)):
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
	parser.add_argument('--classes_count', type=int, dest='classes_count')
	parser.add_argument('--history', type=str, dest='hist_path')
	parser.add_argument('--weights', type=str, dest='weights_path')
	parser.add_argument('--classes', type=str, dest='classes_file')
	args = parser.parse_args()

	sights = {}
	classes = get_ktop_areas(args.classes_count)
	imgs_count = 0
	query = Descriptor.select(Descriptor.image_id, Descriptor.sight_id).tuples()
	for row in tqdm(query):
		if row[1] in classes:
			sights[int(row[0])] = classes.index(row[1])
			imgs_count += 1

	x, y = read_csv(args.fin, sights, False, imgs_count)
	x_nas, y_nas = read_csv(args.fin, sights, True, imgs_count)

#	x, y, classes = areas_cluster(x, y, classes)
	x = np.append(x, x_nas, axis=0)
	y = np.append(y, y_nas, axis=0)

	model = OneVsAllCascade(len(classes), x.shape[1:])
	histories = model.fit(x, y)

	# with open(args.hist_file, 'w') as f:
	# 	f.write(json.dumps(history.history))
	with open(args.classes_file, 'w') as f:
		f.write(json.dumps(classes))
	model.save_weights(args.weights_path)

	for history in histories:
		plot_history(history.history)

	img = load_img('images/rome1.jpg')
	descr = Encoder().get_descriptor(img)

	pred = model.predict(descr, 0.8)
	print(pred)
	sight_id = classes[pred]
	print(sight_id)
	# show_image(args.image_path)
	# look_sight(sight_id)