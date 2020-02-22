import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions, MobileNet
from keras.applications.mobilenet import mobilenet
import numpy as np
import time


def load_img(img_path):
	img = image.load_img(img_path, target_size=(224,224))
	img = image.img_to_array(img)
	return img


class Encoder:
	def __init__(self):
		model = MobileNet(include_top=False, weights='imagenet')

		x = model.output
		x = mobilenet.layers.GlobalAveragePooling2D()(x)

		self.model = mobilenet.models.Model(inputs=model.input, outputs=x)
		self.model.compile(optimizer='sgd', loss='categorical_crossentropy')

	def get_descriptor(self, x, batch=False):
		if not batch:
			x = np.expand_dims(x, axis=0)

		preds = self.model.predict(preprocess_input(x))
		if batch:
			return preds
		else:
			return preds[0]


def test(img_path):
	pre_t = time.time()

	img = load_img(img_path)

	en = Encoder()
	preds = en.get_descriptor(img, False)

	print(preds.shape)
	cur_t = time.time()
	print(cur_t - pre_t)

if __name__ == '__main__':
	test('images/cat.jpg')

