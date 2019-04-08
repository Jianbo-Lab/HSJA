from __future__ import absolute_import, division, print_function 
import tensorflow as tf
import numpy as np
import os
from keras.layers import Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Convolution2D, BatchNormalization, Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute, GlobalAveragePooling2D 
from keras.preprocessing import sequence
from keras.datasets import imdb, mnist
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras import backend as K  
from keras import optimizers
import math

def construct_original_network(dataset_name, model_name, train): 
	data_model = dataset_name + model_name
	
	# Define the model 
	input_size = 32
	num_classes = 10
	channel = 3

	assert model_name == 'resnet'
	from resnet import resnet_v2, lr_schedule,  lr_schedule_sgd
	
	model, image_ph, preds = resnet_v2(input_shape=(input_size, input_size,
	 channel), depth=20, num_classes = num_classes)

	optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)


	model.compile(loss='categorical_crossentropy',
			  optimizer=optimizer,
			  metrics=['accuracy'])

	grads = []
	for c in range(num_classes):
		grads.append(tf.gradients(preds[:,c], image_ph))

	grads = tf.concat(grads, axis = 0)
	approxs = grads * tf.expand_dims(image_ph, 0)

	logits = [layer.output for layer in model.layers][-2]
	print(logits)
		
	sess = K.get_session()

	return image_ph, preds, grads, approxs, sess, model, num_classes, logits

class ImageModel():
	def __init__(self, model_name, dataset_name, train = False, load = False, **kwargs):
		self.model_name = model_name
		self.dataset_name = dataset_name
		self.data_model = dataset_name + model_name
		self.framework = 'keras'

		print('Constructing network...')
		self.input_ph, self.preds, self.grads, self.approxs, self.sess, self.model, self.num_classes, self.logits = construct_original_network(self.dataset_name, self.model_name, train = train)


		self.layers = self.model.layers
		self.last_hidden_layer = self.model.layers[-3]

		self.y_ph = tf.placeholder(tf.float32, shape = [None, self.num_classes])
		if load:
			if load == True:
				print('Loading model weights...')
				self.model.load_weights('{}/models/original.hdf5'.format(
					self.data_model), by_name=True)
			elif load != False:
				self.model.load_weights('{}/models/{}.hdf5'.format(
					self.data_model, load), by_name=True)

	def predict(self, x, verbose=0, batch_size = 500, logits = False):
		x = np.array(x)
		if len(x.shape) == 3:
			_x = np.expand_dims(x, 0) 
		else:
			_x = x
			
		if not logits:
			prob = self.model.predict(_x, batch_size = batch_size, 
			verbose = verbose)
		else:
			num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))
			probs = []
			for i in range(num_iters):
				x_batch = _x[i * batch_size: (i+1) * batch_size]

				prob = self.sess.run(self.logits, 
					feed_dict = {self.input_ph: x_batch})

				probs.append(prob)
				
			prob = np.concatenate(probs, axis = 0)

		if len(x.shape) == 3:
			prob = prob.reshape(-1)

		return prob












