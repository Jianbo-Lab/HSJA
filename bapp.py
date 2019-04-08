from __future__ import absolute_import, division, print_function
import numpy as np

def bapp(model, 
	sample, 
	up_th = 1, 
	low_th = 0, 
	constraint = 'l2', 
	num_iters = 40, 
	gamma = 0.01, 
	target_label = None, 
	target_image = None, 
	epsilon_type = 'geometric_progression', 
	max_batch_size = 1e4,
	init_batch_size = 100):
	"""
	Main algorithm for Boundary Attack ++.

	Inputs:
	model: the object that has predict method. 

	predict outputs probability scores.

	up_th: upper bound of the image.

	lower_th: lower bound of the image.

	constraint: choose between [l2, linf].

	num_iters: number of iterations.

	gamma: used to set binary search threshold theta.

	target_label: integer or None for nontargeted attack.

	target_image: an array with the same size as sample, or None. 

	epsilon_type: choose between 'geometric_progression', 'grid_search'.

	max_batch_size: maximum batch size for estimating gradient.

	init_batch_size: initial batch size for estimating gradient.

	Output:
	perturbed image.
	
	"""
	# Set parameters
	h, w, c = sample.shape
	original_label = np.argmax(model.predict(sample))
	params = {'up_th': up_th, 'low_th': low_th, 'h': h, 'w': w, 'c': c, 
				'original_label': original_label, 
				'target_label': target_label,
				'target_image': target_image, 
				'constraint': constraint,
				'num_iters': num_iters, 
				'gamma': gamma, 
				'd': np.prod(sample.shape), 
				'epsilon_type': epsilon_type,
				'max_batch_size': max_batch_size,
				'init_batch_size': init_batch_size,
				}

	# Set binary search threshold.
	if params['constraint'] == 'l2':
		params['theta'] = params['gamma'] / np.sqrt(params['d'])
	else:
		params['theta'] = params['gamma'] / (params['d'])
		
	# Initialize.
	perturbed = initialize(model, sample, params)
	dist_post_update = compute_distance(perturbed, sample, constraint)

	# Project the initialization to the boundary.
	perturbed, dist = line_search_batch(sample, 
		np.expand_dims(perturbed, 0), 
		model, 
		params)

	for j in np.arange(params['num_iters']):
		params['cur_iter'] = j + 1

		# Choose delta.
		delta = select_delta(params, dist_post_update)

		# Choose batch size.
		batch_size = int(params['init_batch_size'] * np.sqrt(j+1))
		batch_size = int(min([batch_size, params['max_batch_size']]))

		# approximate gradient.
		gradf = approximate_gradient(model, perturbed, batch_size, 
			delta, params)
		if params['constraint'] == 'linf':
			update = np.sign(gradf)
		else:
			update = gradf

		# search step size.
		if params['epsilon_type'] == 'geometric_progression':
			# find step size.
			epsilon = geometric_progression_for_stepsize(perturbed, 
				update, dist, model, params)

			# Update the sample. 
			perturbed = clip_image(perturbed + epsilon * gradf, 
				low_th, up_th)

			# Binary search to return to the boundary. 
			perturbed, dist_post_update = line_search_batch(sample, 
				perturbed[None], model, params)

		elif params['epsilon_type'] == 'grid_search':
			# Grid search for stepsize.
			epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
			perturbeds = x + epsilons.reshape(-1,1,1,1) * update
			perturbeds = clip_image(perturbeds, 
				params['low_th'], params['up_th'])
			labels = np.argmax(model.predict(perturbeds), axis = 1)

			if params['target_label'] is None:
				idx_perturbed = labels != original_label
			else:
				idx_perturbed = labels == params['target_label']

			if np.sum(idx_perturbed) == 0:
				# Do not perturb if all perturbation lies 
				# on the other side of the boundary. 
				perturbed = x
				epsilon_selected = 0
			else:
				# Select the perturbation that yields the minimum distance # after binary search.
				perturbed, dist_post_update = line_search_batch(sample, 
					perturbeds[idx_perturbed], model, params)

		# compute new distance.
		dist = compute_distance(perturbed, sample, constraint)
		print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))

	return perturbed

def clip_image(image, low_th, up_th):
	# Clip an image, or an image batch, with upper and lower threshold.
	return np.minimum(np.maximum(low_th, image), up_th) 


def compute_distance(x_ori, x_pert, constraint = 'l2'):
	# Compute the distance between two images.
	if constraint == 'l2':
		return np.linalg.norm(x_ori - x_pert)
	elif constraint == 'linf':
		return np.max(abs(x_ori - x_pert))


def approximate_gradient(model, sample, num_samples, delta, params):
	up_th, low_th = params['up_th'], params['low_th']
	h,w,c = params['h'], params['w'], params['c']

	# Generate random vectors.
	if params['constraint'] == 'l2':
		rv = np.random.randn(num_samples, h, w, c)
	elif params['constraint'] == 'linf':
		rv = np.random.uniform(low = -1, high = 1, 
			size = (num_samples, h, w, c))

	rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
	perturbed = sample + delta * rv
	perturbed = clip_image(perturbed, low_th, up_th)
	rv = (perturbed - sample) / delta

	# query the model.
	prob = model.predict(perturbed)
	if params['target_label'] is None:
		fval = np.argmax(prob, axis = 1) != params['original_label'] 
		# 1 if label changes. 

	else:
		fval = np.argmax(prob, axis = 1) == params['target_label']

	fval = 2 * fval.astype(float).reshape(-1,1,1,1) - 1.0

	# Baseline subtraction (when fval differs)
	if np.mean(fval) == 1.0: # label changes. 
		gradf = np.mean(rv, axis = 0)
	elif np.mean(fval) == -1.0: # label not change.
		gradf = - np.mean(rv, axis = 0)
	else:
		fval -= np.mean(fval)
		gradf = np.mean(fval * rv, axis = 0) 

	# Get the gradient direction.
	gradf = gradf / np.linalg.norm(gradf)

	return gradf


def project(original_image, perturbed_images, alphas, constraint):
	if constraint == 'l2':
		alphas = alphas.reshape(-1, 1, 1, 1)
		return (1-alphas) * original_image + alphas * perturbed_images
	elif constraint == 'linf':
		out_images = clip_image(
			perturbed_images, 
			original_image - alphas.reshape(-1,1,1,1), 
			original_image + alphas.reshape(-1,1,1,1)
			)
		return out_images


def _line_search_batch(highs, lows, original_image, perturbed_images, model, 
			thresholds, params):
	""" Recursive helper for Binary search to approach the boundar. """

	# Return when threshold is achieved. 
	if np.max((highs - lows) / thresholds) < 1:
		out_image = project(
			original_image, 
			perturbed_images, 
			highs, 
			params['constraint']
			)
		return out_image

	# projection to mids.
	mids = (highs + lows) / 2.0
	mid_images = project(
		original_image, 
		perturbed_images, 
		mids, 
		params['constraint']
		)

	# Update highs and lows based on model decisions.
	mid_labels = np.argmax(model.predict(mid_images), axis = 1)
	if params['target_label'] is None:
		lows = np.where(params['original_label'] == mid_labels, mids, lows)
		highs = np.where(params['original_label'] != mid_labels, mids, highs)
	else:
		lows = np.where(params['target_label'] != mid_labels, mids, lows)
		highs = np.where(params['target_label'] == mid_labels, mids, highs)

	return _line_search_batch(highs, lows, original_image, perturbed_images, 
		model, thresholds, params)


def line_search_batch(original_image, perturbed_images, model, params):
	""" Binary search to approach the boundar. """

	# Compute distance between each of perturbed image and original image.
	dists_post_update = np.array([
			compute_distance(
				original_image, 
				perturbed_image, 
				params['constraint']
			) 
			for perturbed_image in perturbed_images])

	# Choose upper thresholds in binary searchs based on constraint.
	if params['constraint'] == 'linf':
		highs = dists_post_update
		# Stopping criteria.
		thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
	else:
		highs = np.ones(len(perturbed_images))
		thresholds = params['theta']

	lows = np.zeros(len(perturbed_images))

	

	# Call recursive function. 
	out_images = _line_search_batch(highs, lows, original_image, 
		perturbed_images, model, thresholds, params)

	# Compute distance of the output image to select the best choice. 
	# (only used when epsilon_type is grid_search.)
	dists = np.array([
		compute_distance(
			original_image, 
			out_image, 
			params['constraint']
		) 
		for out_image in out_images])
	idx = np.argmin(dists)

	dist = dists_post_update[idx]
	out_image = out_images[idx]
	return out_image, dist


def initialize(model, sample, params):
	""" 
	Implementation of BlendedUniformNoiseAttack in Foolbox.
	"""
	success = 0
	num_evals = 0

	if params['target_image'] is None:
		# increasing scale if initialization fails.
		num = 1000
		epsilons = np.linspace(0, 1, num=num + 1)[1:]
		while success == 0:
			if num_evals < num:
				epsilon = epsilons[num_evals]
			else:
				epsilon = epsilons[-1]

			random_noise = np.random.uniform(-1, 1, 
				size = (params['h'], params['w'], params['c']))

			initialization = clip_image(
				(1- epsilon) * sample +  epsilon * random_noise, 
				params['low_th'], 
				params['up_th']
				)

			prob = model.predict(initialization)
			success = np.argmax(prob) != params['original_label'] 
			# 1 if label changes.
			num_evals += 1

	else:
		initialization = params['target_image']

	return initialization


def geometric_progression_for_stepsize(x, update, dist, model, params):
	"""
	Geometric progression to search for stepsize.
	Keep decreasing stepsize by half until reaching 
	the desired side of the boundary,
	"""
	epsilon = dist / np.sqrt(params['cur_iter']) 
	def phi(epsilon):
		new = x + epsilon * update
		new = clip_image(new, params['low_th'], params['up_th'])

		label = np.argmax(model.predict(new))

		if params['target_label'] is None:
			success = label != params['original_label']
		else:
			success = label == params['target_label']

		return success

	while not phi(epsilon):
		epsilon /= 2.0

	return epsilon

def select_delta(params, dist_post_update):
	""" 
	Choose the delta at the scale of distance 
	between x and perturbed sample. 

	"""
	if params['cur_iter'] == 1:
		delta = 0.1 * (params['up_th'] - params['low_th'])
	else:
		if params['constraint'] == 'l2':
			delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
		elif params['constraint'] == 'linf':
			delta = params['d'] * params['theta'] * dist_post_update	

	return delta


