from os import path as pathlib
import os
import requests
import csv
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	parser.add_argument('--dir', type=str)

	args = parser.parse_args()
	filename = args.file
	dir = args.dir

	if not pathlib.exists(dir):
		os.mkdir(dir)

	lines = 10
	n = 0
	last_percent = 0

	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			pid = int(row[0])
			secret = row[10]
			server = int(row[11])
			farm = int(row[12])
			url = f'http://farm{farm}.staticflickr.com/{server}/{pid}_{secret}.jpg'

			r = requests.get(url)
			with open(dir + f'/{pid}_{secret}.jpg', 'wb') as f:
				f.write(r.content)

			n += 1
			cp = (n / lines) * 100

			if cp > last_percent + 1:
				cp = int(cp)
				print(f'{cp}%' )

