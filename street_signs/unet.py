import keras
from keras import layers

def conv2d_block(filters):
	def wrapper(x):
		# padding should be 'valid' as in original paper, but I did not find any pos or cons for it.
		# at the same time 'same' yelds concantenate layers without cropping.
		x=layers.Conv2D(filters=filters, kernel_size=3, strides=(1, 1), padding='same',
						kernel_initializer='glorot_uniform')(x)
		x=layers.ReLU()(x)

		return x

	return wrapper


def upsample(filters):
	def wrapper(x, skip=None):
		x=layers.UpSampling2D()(x)
		if skip is not None:
			layers.Concatenate()([x, skip])

		x=conv2d_block(filters)(x)
		x=conv2d_block(filters)(x)
		return x

	return wrapper


def create_unet(input_shape, n_classes):
	skips=[]
	inputs=keras.Input(shape=input_shape)

	# encoder
	x=conv2d_block(64)(inputs)
	x=conv2d_block(64)(x)

	skips.append(x)
	x=layers.MaxPool2D(padding='same')(x)

	x=conv2d_block(128)(x)
	x=conv2d_block(128)(x)

	skips.append(x)
	x=layers.MaxPool2D(padding='same')(x)

	x=conv2d_block(256)(x)
	x=conv2d_block(256)(x)

	skips.append(x)
	x=layers.MaxPool2D(padding='same')(x)

	x=conv2d_block(512)(x)
	x=conv2d_block(512)(x)

	skips.append(x)
	x=layers.MaxPool2D(padding='same')(x)
	# encoder

	x=conv2d_block(1024)(x)
	x=conv2d_block(1024)(x)

	# decoder
	x=upsample(512)(x, skips.pop())
	x=upsample(256)(x, skips.pop())
	x=upsample(128)(x, skips.pop())
	x=upsample(64)(x, skips.pop())
	# decoder

	if n_classes==1:
		head=layers.Activation('sigmoid')
	else:
		head=layers.Activation('softmax')

	x=layers.Conv2D(filters=n_classes, kernel_size=3, strides=(1, 1), padding='same')(x)
	outputs=head(x)

	model=keras.Model(inputs=inputs, outputs=outputs)
	return model
