import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import time

pre_t = time.time()
model = keras.applications.mobilenet.MobileNet(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy')

img_path = 'images/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)

print(decode_predictions(preds))
cur_t = time.time()
print(cur_t - pre_t)