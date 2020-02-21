from os import path as pathlib
import os
import requests
import csv
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in', type=str, dest='fin')
	parser.add_argument('--out', type=str, dest='fout')

	args = parser.parse_args()

	fout = open(args.fout, 'w', newline='')
	writer = csv.writer(fout)

	with open(args.fin, newline='') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			str = [row[2]] + row[18:20]

			if str[1] != '' and str[2] != '':
				writer.writerow(str)
