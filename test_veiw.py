import os
import matplotlib.pyplot as plt
import json
import argparse

from classifier import Classifier, OneVsAllCascade
from mobilenetv1_encoder import Encoder, load_img
from algorithm_for_choosing import get_similar_pos, get_data_about_similar_descrs, get_avg_dist_pos
from utils import show_image, look_sight

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

	Class_model = OneVsAllCascade(len(classes), descr.shape, 'stuff/OneVsAll/weights')

	pred = Class_model.predict(descr, 0.95)
	if pred == len(classes)+1:
		print('Not a sight')
		exit()

	sight_id = classes[pred]


	show_image(args.image_path)
	# print(get_similar_pos(get_data_about_similar_descrs(sight_id, descr, 3), args.metadata))
	# print(get_avg_dist_pos(get_data_about_similar_descrs(sight_id, descr, 3), args.metadata))
	look_sight(sight_id)


if __name__ == '__main__':
	main()
