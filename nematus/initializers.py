'''
Parameter initializers
'''

import numpy

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano_util import floatX

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        if scale == 'glorot_inout':
            scale = numpy.sqrt(2.0 / (nin + nout))
        elif scale == 'glorot_in':
            scale = numpy.sqrt(1.0 / nin)
        elif scale == 'glorot_out':
            scale = numpy.sqrt(1.0 / nout)
        W = scale * numpy.random.randn(nin, nout)
    return W.astype(floatX)

