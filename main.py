from __future__ import absolute_import, division, print_function 


from build_model import ImageModel 
from load_data import ImageData, split_data
from bapp import bapp
import numpy as np
import tensorflow as tf
import sys
import os
import pickle
import argparse
import scipy
import itertools

def construct_model_and_data(args):
	"""
	Load model and data on which the attack is carried out.
	Assign target classes and images for targeted attack.
	"""
	data_model = args.dataset_name + args.model_name
	dataset = ImageData(args.dataset_name)
	x_test, y_test = dataset.x_val, dataset.y_val
	reference = - dataset.x_train_mean
	model = ImageModel(args.model_name, args.dataset_name, 
		train = False, load = True)

	# Split the test dataset into two parts.
	# Use the first part for setting target image for targeted attack.
	x_train, y_train, x_test, y_test = split_data(x_test, y_test, model, 
		num_classes = model.num_classes, split_rate = 0.5, 
		sample_per_class = np.min([np.max([200, args.num_samples // 10 * 3]),
		 1000]))

	outputs = {'data_model': data_model,
				'x_test': x_test,
				'y_test': y_test,
				'model': model,
				'clip_max': 1.0,
				'clip_min': 0.0
				}

	if args.attack_type == 'targeted':
		# Assign target class and image for targeted atttack.
		label_train = np.argmax(y_train, axis = 1)
		label_test = np.argmax(y_test, axis = 1)
		x_train_by_class = [x_train[label_train == i] for i in range(model.num_classes)]
		target_img_by_class = np.array([x_train_by_class[i][0] for i in range(model.num_classes)])
		np.random.seed(0)
		target_labels = [np.random.choice([j for j in range(model.num_classes) if j != label]) for label in label_test]
		target_img_ids = [np.random.choice(len(x_train_by_class[target_label])) for target_label in target_labels]
		target_images = [x_train_by_class[target_labels[j]][target_img_id] for j, target_img_id in enumerate(target_img_ids)]
		outputs['target_labels'] = target_labels
		outputs['target_images'] = target_images

	return outputs


def attack(args):
	outputs = construct_model_and_data(args)
	data_model = outputs['data_model']
	x_test = outputs['x_test']
	y_test = outputs['y_test']
	model = outputs['model']
	clip_max = outputs['clip_max']
	clip_min = outputs['clip_min']
	if args.attack_type == 'targeted':
		target_labels = outputs['target_labels']
		target_images = outputs['target_images']

	for i, sample in enumerate(x_test[:args.num_samples]):
		label = np.argmax(y_test[i])

		if args.attack_type == 'targeted':
			target_label = target_labels[i]
			target_image = target_images[i]
		else:
			target_label = None
			target_image = None

		print('attacking the {}th sample...'.format(i))

		perturbed = bapp(model, 
							sample, 
							clip_max = 1, 
							clip_min = 0, 
							constraint = args.constraint, 
							num_iterations = args.num_iterations, 
							gamma = 0.01, 
							target_label = target_label, 
							target_image = target_image, 
							stepsize_search = args.stepsize_search, 
							max_num_evals = 1e4,
							init_num_evals = 100)

		image = np.concatenate([sample, np.zeros((32,8,3)), perturbed], axis = 1)
		scipy.misc.imsave('{}/figs/{}-{}-{}.jpg'.format(data_model, 
			args.attack_type, args.constraint, i), image)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_name', type = str, 
		choices = ['cifar10'], 
		default = 'cifar10') 

	parser.add_argument('--model_name', type = str, 
		choices = ['resnet'], 
		default = 'resnet') 

	parser.add_argument('--constraint', type = str, 
		choices = ['l2', 'linf'], 
		default = 'l2') 

	parser.add_argument('--attack_type', type = str, 
		choices = ['targeted', 'untargeted'], 
		default = 'untargeted') 

	parser.add_argument('--num_samples', type = int, 
		default = 10) 

	parser.add_argument('--num_iterations', type = int, 
		default = 64) 
	parser.add_argument('--stepsize_search', type = str, 
		choices = ['geometric_progression', 'grid_search'], 
		default = 'geometric_progression')

	args = parser.parse_args()
	dict_a = vars(args)

	data_model = args.dataset_name + args.model_name
	if not os.path.exists(data_model):
		os.mkdir(data_model)
	if not os.path.exists('{}/figs'.format(data_model)):
		os.mkdir('{}/figs'.format(data_model))

	attack(args)
