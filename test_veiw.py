import os
import matplotlib.pyplot as plt
import json
import argparse

from classifier import Classifier, classes_count
from mobilenetv1_encoder import Encoder, load_img
from algorithm_for_choosing import get_similar_pos, get_similar_descrs
from utils import show_image

import numpy as np


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, dest='image_path')
	parser.add_argument('--metadata', type=str, dest='metadata')
	parser.add_argument('--classes', type=str, dest='fclasses')
	args = parser.parse_args()

	img = load_img(args.image_path)
	descr = Encoder().get_descriptor(img)

	Class_model = Classifier(classes_count, descr.shape, weights_file='stuff/weights.h5')

	pred = np.argmax(Class_model.predict(descr))
	with open(args.fclasses, "r") as read_file:
		classes = json.load(read_file)
	sight_id = classes[pred]

	show_image(args.image_path)
	print(get_similar_pos(get_similar_descrs(sight_id, descr, 3), args.metadata))


if __name__ == '__main__':
	main()

