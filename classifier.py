import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from keras import layers
from keras import models
from keras import regularizers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from evolutionary_search import EvolutionaryAlgorithmSearchCV


from mobilenetv1_encoder import Encoder, load_img
from sql_models import db_handler, Sight, Descriptor
from algorithm_for_choosing import split_by_labels, unite_by_labels, sight_cluster, score_clustering
from utils import plot_history, get_ktop_areas
import peewee
import csv
import json
from tqdm import tqdm
import argparse

import numpy as np


def sight_classifier(cnt_classes, input_shape, weights_file=None, activation='relu', units=40, hidden_layers=2,
					 dropout=0.005, regularizer=regularizers.l2, regularizer_lambda=0.006, optimizer=optimizers.SGD,
					 lrate=0.004):
	model = models.Sequential()

	for i in range(hidden_layers):
		params = dict(units=units, activation=activation, kernel_regularizer=regularizer(regularizer_lambda))
		if i == 0:
			params['input_shape'] = input_shape

		model.add(layers.Dense(**params))
		model.add(layers.Dropout(dropout))

	if cnt_classes == 2:
		model.add(layers.Dense(units=1, activation='sigmoid'))
	else:
		model.add(layers.Dense(units=cnt_classes, activation='softmax'))

	if weights_file:
		model.load_weights(weights_file)

	if cnt_classes == 2:
		loss = 'binary_crossentropy'
	else:
		loss = 'categorical_crossentropy'

	model.compile(optimizer=optimizer(lrate), loss=loss, metrics=['accuracy'])
	return model


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
		print('%f clusters on %f' % (count, classes[i]))
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


def load_prepare_data(classes_count, binary):
	sights = {}
	classes = get_ktop_areas(classes_count)
	query = Descriptor.select(Descriptor.image_id, Descriptor.sight_id).tuples()
	for row in tqdm(query):
		if row[1] in classes:
			sights[int(row[0])] = classes.index(row[1])

	x, y = read_csv(args.fin, sights, False, len(sights))

	if binary:
		y = np.full_like(y, 0)
		classes = [1]
		x_nas, y_nas = read_csv(args.fin, sights, True, len(sights))
		y_nas = np.full_like(y_nas, 1)
	else:
		x_nas, y_nas = read_csv(args.fin, sights, True, len(sights) // len(classes))
		# x, y, classes = areas_cluster(x, y, classes)
		y_nas = np.full_like(y_nas, np.max(y)+1)

	x = np.append(x, x_nas, axis=0)
	y = np.append(y, y_nas, axis=0)
	classes.append(-1)

	control_indices = np.random.choice(x.shape[0], x.shape[0] // 10)
	c_x, c_y = x[control_indices], y[control_indices]
	x, y = np.delete(x, control_indices, 0), np.delete(y, control_indices, 0)

	rng_state = np.random.get_state()
	np.random.shuffle(x)
	np.random.set_state(rng_state)
	np.random.shuffle(y)

	return x, y, c_x, c_y, classes


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv', type=str, dest='fin')
	parser.add_argument('--classes_count', type=int, dest='classes_count')
	parser.add_argument('--history', type=str, dest='hist_file')
	parser.add_argument('--weights', type=str, dest='weights_file')
	parser.add_argument('--classes', type=str, dest='classes_file')
	args = parser.parse_args()

	par_grid_full = {
		'activation': ['relu', 'tanh'],
		'units': [30, 50, 70],
		'hidden_layers': [1,2,3],
		'dropout': [0.003, 0.005, 0.007, 0.01],
		'regularizer': [regularizers.l1, regularizers.l2],
		'regularizer_lambda': [0.002, 0.005],

		'optimizer': [optimizers.Nadam, optimizers.SGD],
		'lrate': [0.006, 0.01, 0.015, 0.02],

		'batch_size': [20, 50, 100, 300],
		'epochs': [150]
	}
	par_grid_test = {
		'activation': ['relu'],
		'units': [30],
		'hidden_layers': [2],
		'dropout': [0.003],
		'regularizer': [regularizers.l1],
		'regularizer_lambda': [0.002],

		'optimizer': [optimizers.Nadam],
		'lrate': [0.006],

		'batch_size': [20],
		'epochs': [150]
	}

	x, y, c_x, c_y, classes = load_prepare_data(args.classes_count, False)

	#grid = GridSearchCV(KerasClassifier(classifier_model, cnt_classes=len(classes), input_shape=x.shape[1:], verbose=0), par_grid, n_jobs=4, cv=5, verbose=3)
	grid = EvolutionaryAlgorithmSearchCV(estimator=KerasClassifier(sight_classifier, cnt_classes=2, input_shape=x.shape[1:], verbose=0), refit=True,
										 params=par_grid_test, scoring="accuracy", cv=2,
										 verbose=2, population_size=2, gene_mutation_prob=0.08, gene_crossover_prob=0.5,
										 tournament_size=3, generations_number=1, n_jobs=2)
	grid.fit(x, y)
	print('Best: %f using %s' % (grid.best_score_, grid.best_params_))
	print('Control on %f samples' % (len(c_x)))
	c_y_pred = grid.best_estimator_.predict(c_x)

	cm = confusion_matrix(c_y, c_y_pred)
	print(cm)

	#
	# with open(args.hist_file, 'w') as f:
	# 	f.write(json.dumps(history.history))
	# with open(args.classes_file, 'w') as f:
	# 	f.write(json.dumps(classes))
	# model.model.save_weights(args.weights_file)
	# plot_history(history.history)
