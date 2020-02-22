import os
import csv
import argparse
from peewee import *

from sql_models import db_handler, Descriptor

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in', type=str, dest='fin')

	args = parser.parse_args()


	db_handler.connect()
	if not db_handler.table_exists(Descriptor.db_table):
		Descriptor.create_table()


	with open(args.fin, newline='') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			image_id = row[0]
			descr = row[1:]
			row = Descriptor(image_id=image_id, descriptor=','.join(descr))
			row.save()

