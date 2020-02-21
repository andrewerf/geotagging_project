from os import path as pathlib
import os
import requests
import csv
import argparse

from utils import Area, convert_accuracy

accuracy = 200

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in', type=str, dest='fin')
	parser.add_argument('--dir', type=str, dest='fout')

	args = parser.parse_args()

	groups = []
	with open(args.fin, newline='') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			pos = tuple(map(float, row[1:3]))
			area = Area(pos, convert_accuracy(pos, accuracy))
			print(area)

