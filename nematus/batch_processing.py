
'''
Batch processing
'''

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import json
import numpy
import copy
import argparse

import os
import sys
import time
import logging

import itertools

from util import *
from theano_util import *

class Batch(object):

  def __init__(self, seqs_x, seqs_y, weights=None, maxlen=None, n_words_src=30000,
                 n_words=30000, n_factors=1, multi_sentences=False):
    '''
    Batch preparation
    '''

    self.weights = weights
    self.x, self.x_mask = None, None
    self.y, self.y_mask = None, None

    self.n_samples = len(seqs_x)

    if multi_sentences:
        self.main_seqs_x, self.main_seqs_y = [], []
        self.extra_seqs_x, self.extra_seqs_y = [], []

        self.x_extra_ids = []
        self.y_extra_ids = []
        for i in xrange(self.n_samples):
            self.main_seqs_x.append(seqs_x[i][0])
            self.extra_seqs_x.extend(seqs_x[i][1:])
            self.x_extra_ids.extend([i] * (len(seqs_x[i])-1))

            self.main_seqs_y.append(seqs_y[i][0])
            self.extra_seqs_y.extend(seqs_y[i][1:])
            self.y_extra_ids.extend([i] * (len(seqs_y[i])-1))

        seqs_x = self.main_seqs_x + self.extra_seqs_x
        seqs_y = self.main_seqs_y + self.extra_seqs_y

        self.x_extra_ids = numpy.array(self.x_extra_ids).astype('int64')
        self.y_extra_ids = numpy.array(self.y_extra_ids).astype('int64')
    else:
        self.main_seqs_x = seqs_x
        self.main_seqs_y = seqs_y
        self.extra_seqs_x, self.extra_seqs_y = [], []
        self.x_extra_ids, self.y_extra_ids = None, None

    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    
        

#    if maxlen is not None:
#        new_seqs_x = []
#        new_seqs_y = []
#        new_lengths_x = []
#        new_lengths_y = []
#        new_weights = []
#        if weights is None:
#            weights = [None] * len(seqs_y) # to make the zip easier
#        for l_x, s_x, l_y, s_y, w in zip(lengths_x, seqs_x, lengths_y, seqs_y, weights):
#            if l_x < maxlen and l_y < maxlen:
#                new_seqs_x.append(s_x)
#                new_lengths_x.append(l_x)
#                new_seqs_y.append(s_y)
#                new_lengths_y.append(l_y)
#                new_weights.append(w)
#        lengths_x = new_lengths_x
#        seqs_x = new_seqs_x
#        lengths_y = new_lengths_y
#        seqs_y = new_seqs_y
#        weights = new_weights
#
#        if len(lengths_x) < 1 or len(lengths_y) < 1:
#            if weights is not None:
#                return None, None, None, None, None
#            else:
#                return None, None, None, None
#

    self.batch_size_x = len(seqs_x)
    self.batch_size_y = len(seqs_y)

    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    self.x = numpy.zeros((n_factors, maxlen_x, self.batch_size_x)).astype('int64')
    self.y = numpy.zeros((maxlen_y, self.batch_size_y)).astype('int64')
    self.x_mask = numpy.zeros((maxlen_x, self.batch_size_x)).astype(floatX)
    self.y_mask = numpy.zeros((maxlen_y, self.batch_size_y)).astype(floatX)
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        self.x[:, :lengths_x[idx], idx] = zip(*s_x)
        self.x_mask[:lengths_x[idx]+1, idx] = 1.
        self.y[:lengths_y[idx], idx] = s_y
        self.y_mask[:lengths_y[idx]+1, idx] = 1.

#    if weights is not None:
#        return x, x_mask, y, y_mask, weights
#    else:
#        return x, x_mask, y, y_mask

