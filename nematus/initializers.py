'''
Parameter initializers
'''

import numpy

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def uniform_weight(nin, nout=None, scale=1.0, use_glorot=True):
    if nout is None:
        nout = nin
    if use_glorot:
        scale *= numpy.sqrt(6.0 / (nin + nout))
    W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')
