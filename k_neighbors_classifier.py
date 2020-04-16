from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pylab as pl
import json


classes_count = 20


def split_data(data, percent=10, count=-1):
	if count != -1:
		data = data[:count]
	train_cnt = int(len(data[0]) * ((100 - percent) / 100))
	return data[0][:train_cnt], data[0][train_cnt:], data[1][:train_cnt], data[1][train_cnt:]


def get_data(data_path):
	with open(data_path, "r") as read_file:
		data = json.load(read_file)
	x = list()
	y = list()
	for i in data:
		x.append(data[i]['descr'])
		y.append(data[i]['label'])
	return x, y


def collect_data(data_output_path):
	with open(data_output_path, "w") as write_file:
		data = dict()
		x, y = get_data(args.csv_path)
		for i in range(len(x)):
			data[i] = {'descr': list(x[i]), 'label': int(y[i])}
		json.dump(data, write_file)


def show_data(n_classes=classes_count, d_data=None):
	if d_data is None:
		d_data = list()
	x_train = d_data[0]
	y_train = d_data[1]
	pca = PCA()
	reduced = pca.fit_transform(x_train)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=y_train)
	pl.show()


def mahalanobis_dist(a, b):
	xx = a.T
	yy = b.T
	e = xx - yy
	X = np.vstack([xx, yy])
	V = np.cov(X.T)
	if np.linalg.det(V) == 0:
		p = np.eye(V.shape[0])
	else:
		p = np.linalg.inv(V)
	D = np.sqrt(np.sum(np.dot(e, p) * e))
	return D


def classifier(csv_path):
	a = np.array([1, 5, 8, 7])
	b = np.array([4, 56, 367, 5])
	print(mahalanobis_dist(a, b))


class AutoClassifiers:

	def __init__(self, data_path, graph_path, json_output_path, metric=None, limit=None):
		self.data_path = data_path
		self.graph_path = graph_path
		self.json_output_path = json_output_path
		self.metric = metric
		self.classes_count = classes_count
		self.limit = limit
		self.classification_results = dict()

	def fit_models(self):
		data = get_data(self.data_path)
		x_train, x_test, y_train, y_test = split_data(data, percent=15, count=10000)
		if self.limit is not None:
			x_train = x_train[:self.limit]
			x_test = x_test[:self.limit]
			y_train = y_train[:self.limit]
			y_test = y_test[:self.limit]
			data = data[0][:self.limit]
		else:
			data = data[0]
		neighbors = np.arange(2, self.classes_count + 1)
		train_accuracy = np.empty(len(neighbors))
		test_accuracy = np.empty(len(neighbors))
		for i, k in tqdm(enumerate(neighbors)):
			if self.metric is None:
				knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
			elif self.metric == 'mahalanobis':
				knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='mahalanobis', metric_params={'V': np.cov(x_train)}, n_jobs=-1)
			else:
				knn = KNeighborsClassifier(n_neighbors=k, metric=self.metric, n_jobs=-1)
			knn.fit(x_train, y_train)
			train_accuracy[i] = knn.score(x_train, y_train)
			test_accuracy[i] = knn.score(x_test, y_test)
			neighbors_res = knn.kneighbors(data, n_neighbors=k)
			self.classification_results[str(k)] = list(zip(list(map(lambda x: list(x), neighbors_res[0])), list(
				map(lambda x: list(map(lambda y: int(y), list(x))), neighbors_res[1]))))

		plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
		plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
		plt.legend()
		plt.xlabel('n_neighbors')
		plt.ylabel('Accuracy')
		plt.savefig(self.graph_path)

	def get_models_results(self):
		with open(self.json_output_path, "w") as write_file:
			json.dump(self.classification_results, write_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, dest='data_path')
	parser.add_argument('--graph_path', type=str, dest='g_path')
	parser.add_argument('--json_output_path', type=str, dest='json_output_path')

	args = parser.parse_args()

	clustering_model = AutoClassifiers(data_path=args.data_path, json_output_path=args.json_output_path, graph_path=args.g_path, metric='mahalanobis', limit=50)
	clustering_model.fit_models()
	clustering_model.get_models_results()
