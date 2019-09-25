# Untilities for Minimum Risk Training

import tensorflow as tf
import numpy as np
import inference
import itertools
import util
import math
from metrics.scorer_provider import ScorerProvider


def full_sampler(replica, sess, config, x, x_mask, y, y_mask, init_translation_maxlen):
	"""generate candidate sentences used for Minimum Risk Training

	Args:
		replica: inference models to do sampling
		x: (factor, len, batch_size)
		x_mask: (len, batch_size)
		y: (len, batch_size)
		y_mask: (len, batch_size)
		init_translation_maxlen: pre-set maximum translation length
	Returns:
		x, x_mask, y, y_mask are four lists containing the corresponding content of
		source-candidate sentence pairs, with shape:
		x: (factor, len, batch_size*sampleN)
		x_mask: (len, batch_size*sampleN)
		y: (len, batch_size*sampleN)
		y_mask: (len, batch_size*sampleN)

		y is a list of the corresponding references; index is
		a list of number indicating the starting point of different source sentences.
	"""

	sampleN = config.samplesN
	# expand input for parallel multiple samplings
	x_new = np.repeat(x, sampleN, axis=2)
	x_mask_new = np.repeat(x_mask, sampleN, axis=1)

	if config.sample_way == 'beam_search':

		# set inference model
		model_set = inference.InferenceModelSet([replica], [config])

		# set maximum number of tokens of sampled candidates
		trans_maxlen = int(config.max_len_a * x_mask.shape[0] + config.max_len_b)
		if trans_maxlen < init_translation_maxlen:
			config.translation_maxlen = trans_maxlen
		else:
			config.translation_maxlen = init_translation_maxlen

		# split the minibatch into multiple sub-batches, and execute samplings for each sub-batch separately
		if config.max_sentences_of_sampling > 0:
			# number of split equals to batch_size / maximum accepted sentences for sampling (in a device)
			num_split = math.ceil(x_mask.shape[1] / config.max_sentences_of_sampling)
			# split the numpy array into a list of numpy array
			split_x = np.array_split(x, num_split, 2)
			split_x_mask = np.array_split(x_mask, num_split, 1)
			sample_and_score = []
			# feed sub-batch into model to generate samples
			for i in range(len(split_x)):
				sample_and_score += model_set.beam_search(sess, split_x[i], split_x_mask[i],
										beam_size=config.samplesN,
										normalization_alpha=config.normalization_alpha)
		else:
			sample_and_score = model_set.beam_search(sess, x, x_mask,
											beam_size=config.samplesN,
											normalization_alpha=config.normalization_alpha)
		# sample_and_score: outer: batch_size, inner: sampleN elements(each represents a sample)

		# fetch samplings
		samples=[]
		for i, ss in enumerate(sample_and_score):
			samples.append([])
			for (sample_seq, cost) in ss:
				samples[i].append(sample_seq.tolist())
		# samples: list with shape (batch_size, sampleN, len), uneven
		# beam search sampling, no need to remove duplicate samples.

		# samples number of each batch (useless in beam sampling mode)
		index = [[0]]

	elif config.sample_way == 'randomly_sample':

		model_set = inference.InferenceModelSet([replica], [config])

		trans_maxlen = int(config.max_len_a*x_mask.shape[0]+config.max_len_b)
		if trans_maxlen < init_translation_maxlen:
			config.translation_maxlen = trans_maxlen
		else:
			config.translation_maxlen = init_translation_maxlen

		samples = []
		for i in range(x_mask.shape[1]):
			samples.append([])

		# at the moment, it is not necessary to set this value when doing randomly sampling
		# but the codes could be used after optimisation
		if config.max_sentences_of_sampling > 0:
			num_split = math.ceil(x_mask_new.shape[1] / config.max_sentences_of_sampling)
			split_x = np.array_split(x_new, num_split, 2)
			split_x_mask = np.array_split(x_mask_new, num_split, 1)
			sample = model_set.sample(sess, split_x[0], split_x_mask[0])
			for i in range(1, len(split_x)):
				sample = np.concatenate((sample, model_set.sample(sess, split_x[i], split_x_mask[i])))
		else:
			sample = model_set.sample(sess, x_new, x_mask_new)
		# sample: a list of NumPy arrays (each numpy array contains a sampling a source sentence in x_new).
		for i in range(len(sample)):
			for ss in sample[i*sampleN:(i+1)*sampleN]:
				samples[i].append(ss.tolist())
			# samples: list with shape (batch_size, sampleN, len), uneven

		# remove duplicate samples
		for i in range(len(samples)):
			samples[i].sort()
			samples[i] = [s for s, _ in itertools.groupby(samples[i])]

		# remove the corresponding x and x_mask
		index = []
		for i in range(len(samples)):
			index.append(len(samples[i]))
		for i in range(x_mask.shape[1]):
			x_new = np.repeat(x, index, axis=2)
			x_mask_new = np.repeat(x_mask, index, axis=1)

		# calculate the the number of remaining candidate samplings for each source sentence,
		# store the information in 'index' for the subsequent normalisation of distribution and calculation of
		# expected risk.
		index = [[0]]
		for i in range((len(samples))):
			index[0].append(index[0][i] + len(samples[i]))

	else:
		assert False

	# add reference in candidate sentences:

	# convert from time domain to batch domain
	y = list(map(list, zip(*y)))
	# y: batch_size X len
	y_mask = list(map(list, zip(*y_mask)))
	if config.mrt_reference:
		for i in range(len(samples)):
			# delete the pad of reference
			lenth = int(sum(y_mask[i]))
			y[i] = y[i][:lenth]
			# reference always at the first
			if y[i] not in samples[i]:
				samples[i].append(y[i])
				samples[i].pop(-2)

	# add padding: (no specific padding token, just assign 0(<EOS>) and masked to avoid generating loss)

	# combine samples from different batches (decrease the outermost dimension)
	ss = []
	for i in samples:
		ss += i
	samples = ss
	# samples: list with shape (batch_size*sampleN, len), uneven
	n_samples = len(samples)
	lengths_y = [len(s) for s in samples]
	maxlen_y = np.max(lengths_y) + 1

	y_new = np.zeros((maxlen_y, n_samples)).astype('int64')
	y_mask_new = np.zeros((maxlen_y, n_samples)).astype('float32')

	for idx, s_y in enumerate(samples):
		y_new[:lengths_y[idx], idx] = s_y
		y_mask_new[:lengths_y[idx] + 1, idx] = 1.

	return x_new.tolist(), x_mask_new.tolist(), y_new.tolist(), y_mask_new.tolist(), y, index


def cal_metrics_score(samples, config, num_to_target, refs, index):
	"""evaluate candidate sentences based on reference with evaluation metrics
	Args:
		samples: candidate sentences in list (with padding) (maxlen, batch_size*sampleN)
		num_to_target: dictionary to map number to word
		refs: ground truth translations in list (batch_size, len), uneven
		index: starting point of each source sentence
	Return:
		numpy array contains scores of candidates
	"""

	samplesN = config.samplesN
	batch_size = len(refs)

	# convert from time domain to batch domain
	samples = list(map(list, zip(*samples)))
	samples_totalN = len(samples)

	if config.sample_way == 'beam_search':
		scores = np.zeros((batch_size * samplesN)).astype('float32')
		for i in range(int(batch_size)):
			ref = util.seq2words(refs[i], num_to_target).split(" ")

			ss = []
			for j in samples[i * samplesN:(i + 1) * samplesN]:
				ss.append(util.seq2words(j, num_to_target))
			ss = [s.split(" ") for s in ss]
			# ss: list with (samplesN, len), uneven(seq2word could get rid of padding)

			# get evaluation metrics (negative smoothed BLEU) for samplings
			scorer = ScorerProvider().get(config.mrt_loss)
			scorer.set_reference(ref)
			score = np.array(scorer.score_matrix(ss))
			# compute the negative BLEU score (use 1-BLEU (BLEU: 0~1))
			scores[i * samplesN:(i + 1) * samplesN] = 1 - 1 * score
	else:
		# for randomly sampling strategy, starting point information needed
		scores = np.zeros((samples_totalN)).astype('float32')
		for i in range(int(batch_size)):
			ref = util.seq2words(refs[i], num_to_target).split(" ")

			ss = []
			for j in samples[index[0][i]: index[0][i+1]]:
				ss.append(util.seq2words(j, num_to_target))
			ss = [s.split(" ") for s in ss]

			# get negative smoothed BLEU for samples
			scorer = ScorerProvider().get(config.mrt_loss)
			scorer.set_reference(ref)
			score = np.array(scorer.score_matrix(ss))
			# compute the negative BLEU score (use 1-BLEU (BLEU: 0~1))
			scores[index[0][i]:index[0][i+1]] = 1 - 1 * score

	return scores


def mrt_cost(cost, score, config):
	"""Calculate expected risk according to evaluation scores and model's translation probability over
	a subset of candidate sentences
	Args:
		cost: translation probabilities, 1D tensor with size: (sub-batch_size, sampleN)
		score: evaluation score, 1D tensor with size: (sub-batch_size, sampleN)
	Return:
		expected risk per real sentence over a sub-batch, a scalar tensor
	"""

	samplesN = config.samplesN
	total_sample = tf.shape(cost)[0]
	batch_size = tf.div(total_sample, samplesN)

	# cancelling the negative of the cost (P**alpha = e**(-alpha*(-logP))
	alpha = tf.constant([-config.mrt_alpha], dtype=tf.float32)
	cost = tf.multiply(cost, alpha)

	# normalise costs
	i = tf.constant(0)

	def while_condition(i, _):
		return tf.less(i, batch_size)
	def body(i, cost):
		# increment i
		normalised_cost = tf.nn.softmax(cost[i * samplesN:(i + 1) * samplesN])
		# assign value of sub-tensor to a tensor iteratively
		if i == 0:
			val = normalised_cost
			part2 = cost[(i + 1) * samplesN:]
			cost = tf.concat([val, part2], axis=0)
		else:
			part1 = cost[:i * samplesN]
			val = normalised_cost
			part2 = cost[(i + 1) * samplesN:]
			cost = tf.concat([part1, val, part2], axis=0)

		return tf.add(i, 1), cost
	# do the loop:
	i, cost = tf.while_loop(while_condition, body, [i, cost])

	# compute expected risk by dot product normalised cost and score
	cost = tf.reshape(cost, [1, total_sample])
	score = tf.reshape(score, [total_sample, 1])
	# calculate the risk per real sentence (before sampling)
	MRTloss = tf.divide(tf.matmul(cost, score)[0][0], tf.cast(batch_size, tf.float32))

	return MRTloss


def mrt_cost_random(cost, score, index, config):
	# basically same as mrt_cost, just need to use index to distinguish different subspaces

	samplesN = config.samplesN
	total_sample = tf.shape(cost)[0]
	batch_size = tf.shape(index)[0] - tf.constant(1)

	# cancelling the negative of the cost
	alpha = tf.constant([-config.mrt_alpha], dtype=tf.float32)
	cost = tf.multiply(cost, alpha)

	# normalise costs
	i = tf.constant(0)

	def while_condition(i, _):
		return tf.less(i, batch_size)
	def body(i, cost):
		normalised_cost = tf.nn.softmax(cost[index[i]: index[i+1]])
		# assign value of sub-tensor to a tensor iteratively
		if i == 0:
			val = normalised_cost
			part2 = cost[index[i+1]:]
			cost = tf.concat([val, part2], axis=0)
		else:
			part1 = cost[:index[i]]
			val = normalised_cost
			part2 = cost[index[i+1]:]
			cost = tf.concat([part1, val, part2], axis=0)

		return tf.add(i, 1), cost
	# do the loop:
	i, cost = tf.while_loop(while_condition, body, [i, cost])

	# compute risk by dot product normalised cost and score
	cost = tf.reshape(cost, [1, total_sample])
	score = tf.reshape(score, [total_sample, 1])
	# calculate the risk per real sentence (before sampling)
	MRTloss = tf.divide(tf.matmul(cost, score)[0][0],tf.cast(batch_size, tf.float32))

	return MRTloss
