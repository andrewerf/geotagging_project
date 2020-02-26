import os
import json, csv
import argparse
from peewee import *
from tqdm import tqdm, trange

from sql_models import db_handler, Sight, Descriptor


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--areas', type=str, dest='areas_file')
	parser.add_argument('--descrs', type=str, dest='descrs_file')
	args = parser.parse_args()

	db_handler.connect()
	Sight.create_table(safe=True)
	Descriptor.create_table(safe=True)

	areas_map = {}
	with open(args.areas_file) as jsonfile:
		sights = json.loads(jsonfile.read())

		for i in trange(len(sights)):
			area = sights[i][0]
			sarea = ','.join(list(map(str, area)))

			row = Sight(area=sarea, type_id=-1)
			row.save()

			for image_id in sights[i][1]:
				areas_map[image_id] = row.id

	with open(args.descrs_file, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for row in tqdm(reader):
			image_id = int(row[0])

			if not image_id in areas_map:
				print('No such image in base: ', image_id)
				continue

			descr = row[1:]
			row = Descriptor(image_id=str(image_id), descriptor=','.join(descr), sight_id=areas_map[image_id])
			row.save()

