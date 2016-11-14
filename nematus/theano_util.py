'''
Theano utility functions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict
import warnings

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# push parameters to Theano shared variables
def zip_to_theano(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip_from_theano(zipped, excluding_prefix=None):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        if excluding_prefix and (kk.startswith(excluding_prefix)):
            continue
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# make prefix-appended name
def pp(pp, name):
    return '%s_%s' % (pp, name)

# initialize Theano shared variables according to the initial parameters
def init_theano_params(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params, with_prefix=''):
    pp = numpy.load(path)
    new_params = OrderedDict()
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        new_params[with_prefix+kk] = pp[kk]

    params.update(new_params)
    return params

def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

# return name of word embedding for factor i
# special handling of factor 0 for backward compatibility
def embedding_name(i):
    if i == 0:
        return 'Wemb'
    else:
        return 'Wemb'+str(i)

# Zero out all parameters
def zero_all(params):
    for kk, vv in params.iteritems():
        vv[:] = numpy.zeros_like(vv)

