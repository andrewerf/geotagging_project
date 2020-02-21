import json
import argparse

from utils import Area, convert_accuracy


def json_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in', type=str, dest='fin')
	parser.add_argument('--dir', type=str, dest='fout')

	args = parser.parse_args()

	pos = []

	with open(args.fin) as json_file:
		data = json.load(json_file)
		for p in data:
			pos.append((int(p['id']),float(p['loc']['lat']), float(p['loc']['lng'])))
	return pos

def group():
	data = json_parse()
	pos = []
	for p in data:
		area = Area((p[1], p[2]))
		pos.append(area)
	grop = []


if __name__ == '__main__':
	group()

