import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from geotagging_project.utils import Area
import peewee
import json
from geotagging_project.sql_models import db_handler, Sight, Descriptor
from geotagging_project.classifier import Classifier, classes_count, get_ktop_areas
from geotagging_project.mobilenetv1_encoder import Encoder, load_img
import numpy as np

imgs_path = '/media/qunity/Workspace/Python_projects/NeuralNetworks/Images'


# if __name__ == '__main__':
# 	pos = (59.948998,30.391515)
# 	a = Area((59.948998,30.395515))
#
# 	print(pos in a)

def show_image(img_path):
	img = plt.imread(img_path)
	plt.imshow(img)
	plt.show()


def show_images(image_id, imgs_path):
	name = image_id + '.jpg'
	for root, dirs, files in os.walk(imgs_path):
		if name in files:
			img = plt.imread(os.path.join(imgs_path, root, name))
			plt.imshow(img)
			plt.show()


def look_sight(sight_id):
	descr_ids = Descriptor.select(Descriptor.id).join(Sight).where(Sight.id == sight_id)

	for id in descr_ids:
		image_id = Descriptor.select(Descriptor.image_id).where(Descriptor.id == id)[0].image_id
		show_images(image_id)


if __name__ == '__main__':
	# img = load_img('images/london1.jpg')
	# descr = Encoder().get_descriptor(img)
	# classes = json.loads(open('stuff/classes.json').read())
	#
	# model = Classifier(classes_count, descr.shape, weights_file='stuff/weights.h5')
	# pred = model.predict(descr)
	# sight_id = classes[np.argmax(pred)]
	# print(sight_id, np.max(pred))

	# look_sight(7775)
	show_images('2970202652')
