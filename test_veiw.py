import os
import matplotlib.pyplot as plt
import json
import argparse

from geotagging_project.classifier import Classifier, classes_count
from geotagging_project.mobilenetv1_encoder import Encoder, load_img
from geotagging_project.algorithm_for_choosing import get_similar_pos, get_data_about_similar_descrs, get_avg_dist_pos
from geotagging_project.utils import show_image, look_sight

import numpy as np


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, dest='image_path')
	parser.add_argument('--metadata', type=str, dest='metadata')
	parser.add_argument('--classes', type=str, dest='fclasses')
	args = parser.parse_args()

	img = load_img(args.image_path)
	descr = Encoder().get_descriptor(img)

	with open(args.fclasses, "r") as read_file:
		classes = json.load(read_file)

	Class_model = Classifier(len(classes), descr.shape, weights_file='stuff/weights2.h5')

	t = Class_model.predict(descr)
	pred = np.argmax(t)
	acc = np.max(t)
	print(t)
	print(acc)

	sight_id = classes[pred]
	if sight_id == -1:
		print('Not a sight')
		exit()

	show_image(args.image_path)
	# print(get_similar_pos(get_data_about_similar_descrs(sight_id, descr, 3), args.metadata))
	print(get_avg_dist_pos(get_data_about_similar_descrs(sight_id, descr, 3), args.metadata))
	# look_sight(14514)


if __name__ == '__main__':
	main()
