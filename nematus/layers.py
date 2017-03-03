'''
Layer definitions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import *
from util import *
from theano_util import *
from alignment_util import *

#from theano import printing

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_reuse_att': ('param_init_gru_cond_reuse_att', 'gru_cond_layer_reuse_att'),
          'gru_local': ('param_init_gru_local', 'gru_local_layer'),
          'embedding': {'param_init_embedding_layer', 'embedding_layer'}
          }


def dropout_constr(options, use_noise, trng, sampling):
    """This constructor takes care of the fact that we want different
    behaviour in training and sampling, and keeps backward compatibility:
    on older versions, activations need to be rescaled at test time;
    on newer veresions, they are rescaled at training time.
    """

    # if dropout is off, or we don't need it because we're sampling, multiply by 1
    # this is also why we make all arguments optional
    def get_layer(shape=None, dropout_probability=0, num=1):
        if num > 1:
            return theano.shared(numpy.array([1.]*num, dtype=floatX))
        else:
            return theano.shared(numpy_floatX(1.))

    if options['use_dropout']:
        # models trained with old dropout need to be rescaled at test time
        if sampling and options['model_version'] < 0.1:
            def get_layer(shape=None, dropout_probability=0, num=1):
                if num > 1:
                    return theano.shared(numpy.array([1-dropout_probability]*num, dtype=floatX))
                else:
                    return theano.shared(numpy_floatX(1-dropout_probability))
        elif not sampling:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            def get_layer(shape, dropout_probability=0, num=1):
                if num > 1:
                    return shared_dropout_layer((num,) + shape, use_noise, trng, 1-dropout_probability, scaled)
                else:
                    return shared_dropout_layer(shape, use_noise, trng, 1-dropout_probability, scaled)

    return get_layer


def get_layer_param(name):
    param_fn, constr_fn = layers[name]
    return eval(param_fn)

def get_layer_constr(name):
    param_fn, constr_fn = layers[name]
    return eval(constr_fn)

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value, scaled=True):
    #re-scale dropout at training time, so we don't need to at test time
    if scaled:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX)/value,
            theano.shared(numpy_floatX(1.)))
    else:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX),
            theano.shared(numpy_floatX(value)))
    return proj

# layer normalization
# code from https://github.com/ryankiros/layer-norm
def layer_norm(x, b, s, norm_only=False):
    _eps = numpy_floatX(1e-5)
    if norm_only:
        x_rawstd = tensor.sqrt((x * x).mean(axis=1, keepdims=True) + _eps)
        output = x / x_rawstd
    else:
        output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
    output = s[None, :] * output + b[None,:]
    return output

def layer_norm3d(x, b, s, norm_only=False):
    _eps = numpy_floatX(1e-5)
    if norm_only:
        x_rawstd = tensor.sqrt((x * x).mean(axis=2, keepdims=True) + _eps)
        output = x / x_rawstd
    else:
        output = (x - x.mean(2)[:,:,None]) / numpy.sqrt((x.var(2)[:,:,None] + _eps))
    output = s[None, None, :] * output + b[None, None,:]
    return output

def weight_norm(W, s):
    """
    Normalize the columns of a matrix
    """
    _eps = numpy_floatX(1e-5)
    W_norms = tensor.sqrt((W * W).sum(axis=0, keepdims=True) + _eps)
    W_norms_s = W_norms * s # do this first to ensure proper broadcasting
    return W / W_norms_s

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, weight_matrix=True, bias=True, followed_by_softmax=False):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    if weight_matrix:
        params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if bias:
       params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype(floatX)

    if options['layer_normalisation_ff'] and not followed_by_softmax:
        scale_add = 0.0
        scale_mul = 1.0
        params[pp(prefix,'ln_b')] = scale_add * numpy.ones((1*nout)).astype(floatX)
        params[pp(prefix,'ln_s')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    if options['weight_normalization'] and not followed_by_softmax:
        scale_mul = 1.0
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    return params


def fflayer(tparams, state_below, options, dropout, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', W=None, b=None, dropout_probability=0, followed_by_softmax=False, **kwargs):
    if W == None:
        W = tparams[pp(prefix, 'W')]
    if b == None:
        b = tparams[pp(prefix, 'b')]

    # for three-dimensional tensors, we assume that first dimension is number of timesteps
    # we want to apply same mask to all timesteps
    if state_below.ndim == 3:
        dropout_shape = (state_below.shape[1], state_below.shape[2])
    else:
        dropout_shape = state_below.shape
    dropout_mask = dropout(dropout_shape, dropout_probability)

    if options['weight_normalization'] and not followed_by_softmax:
         W = weight_norm(W, tparams[pp(prefix, 'W_wns')])
    preact = tensor.dot(state_below*dropout_mask, W) + b

    if options['layer_normalisation_ff'] and not followed_by_softmax:
            if state_below.ndim == 3:
                preact = layer_norm3d(preact, tparams[pp(prefix,'ln_b')], tparams[pp(prefix,'ln_s')], options['layer_normalization_norm_only'])
            else:
                preact = layer_norm(preact, tparams[pp(prefix,'ln_b')], tparams[pp(prefix,'ln_s')], options['layer_normalization_norm_only'])

    return eval(activ)(preact)

# embedding layer
def param_init_embedding_layer(options, params, n_words, dims, factors=None, prefix='', suffix=''):
    if factors == None:
        factors = 1
        dims = [dims]
    for factor in xrange(factors):
        params[prefix+embedding_name(factor)+suffix] = norm_weight(n_words, dims[factor])
    return params

def embedding_layer(tparams, ids, factors=None, prefix='', suffix=''):
    do_reshape = False
    if factors == None:
        if ids.ndim > 1:
            do_reshape = True
            n_timesteps = ids.shape[0]
            n_samples = ids.shape[1]
        emb = tparams[prefix+embedding_name(0)+suffix][ids.flatten()]
    else:
        if ids.ndim > 2:
          do_reshape = True
          n_timesteps = ids.shape[1]
          n_samples = ids.shape[2]
        emb_list = [tparams[prefix+embedding_name(factor)+suffix][ids[factor].flatten()] for factor in xrange(factors)]
        emb = concatenate(emb_list, axis=1)
    if do_reshape:
        emb = emb.reshape((n_timesteps, n_samples, -1))

    return emb

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[pp(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim,)).astype(floatX)

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[pp(prefix, 'Ux')] = Ux

    if options['layer_normalisation']:
        # layer-normalization parameters
        scale_add = 0.0
        scale_mul = 1.0
        params[pp(prefix,'ln_b1')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b2')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_b3')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b4')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s1')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s2')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s3')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s4')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    if options['weight_normalization']:
        scale_mul = 1.0
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def gru_layer(tparams, state_below, options, dropout, prefix='gru',
              mask=None, one_step=False,
              init_state=None,
              dropout_probability_below=0,
              dropout_probability_rec=0,
              truncate_gradient=-1,
              profile=False,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalization']: 
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=2)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'W'))) + \
        tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'Wx'))) + \
        tparams[pp(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, rec_dropout):

        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'ln_b1')], tparams[pp(prefix, 'ln_s1')], options['layer_normalization_norm_only'])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'ln_b2')], tparams[pp(prefix, 'ln_s2')], options['layer_normalization_norm_only'])

        preact = tensor.dot(h_*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact = layer_norm(preact, tparams[pp(prefix, 'ln_b3')], tparams[pp(prefix, 'ln_s3')], options['layer_normalization_norm_only'])
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx = layer_norm(preactx, tparams[pp(prefix, 'ln_b4')], tparams[pp(prefix, 'ln_s4')], options['layer_normalization_norm_only'])
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [rec_dropout]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['layer_normalisation']:
        # layer-normalization parameters
        scale_add = 0.0
        scale_mul = 1.0
        params[pp(prefix,'ln_b1')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b2')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_b3')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b4')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_b5')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b6')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_b7')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_b8')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s1')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s2')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s3')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s4')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s5')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s6')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'ln_s7')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'ln_s8')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    if options['weight_normalization']:
        scale_mul = 1.0
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'U_nl_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Ux_nl_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Wc_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wcx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'U_att_wns')] = scale_mul * numpy.ones((1*1)).astype(floatX)

    return params


def gru_cond_layer(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalization']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=5)
    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) +\
            tparams[pp(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'Wx'))) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'W'))) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout):

        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'ln_b1')], tparams[pp(prefix, 'ln_s1')], options['layer_normalization_norm_only'])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'ln_b2')], tparams[pp(prefix, 'ln_s2')], options['layer_normalization_norm_only'])

        preact1 = tensor.dot(h_*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'ln_b3')], tparams[pp(prefix, 'ln_s3')], options['layer_normalization_norm_only'])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'ln_b4')], tparams[pp(prefix, 'ln_s4')], options['layer_normalization_norm_only'])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], wn(pp(prefix, 'U_att')))+tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1*rec_dropout[3], wn(pp(prefix, 'U_nl')))+tparams[pp(prefix, 'b_nl')]
        if options['layer_normalisation']:
            preact2 = layer_norm(preact2, tparams[pp(prefix, 'ln_b5')], tparams[pp(prefix, 'ln_s5')], options['layer_normalization_norm_only'])
        ctx1_ = tensor.dot(ctx_*ctx_dropout[2], wn(pp(prefix, 'Wc')))
        if options['layer_normalisation']:
            ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'ln_b6')], tparams[pp(prefix, 'ln_s6')], options['layer_normalization_norm_only'])
        preact2 += ctx1_
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], wn(pp(prefix, 'Ux_nl')))+tparams[pp(prefix, 'bx_nl')]
        if options['layer_normalisation']:
            preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'ln_b7')], tparams[pp(prefix, 'ln_s7')], options['layer_normalization_norm_only'])
        preactx2 *= r2
        ctx2_ = tensor.dot(ctx_*ctx_dropout[3], wn(pp(prefix, 'Wcx')))
        if options['layer_normalisation']:
            ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'ln_b8')], tparams[pp(prefix, 'ln_s8')], options['layer_normalization_norm_only'])
        preactx2 += ctx2_

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)
    return rval



# Conditional GRU layer with Attention (but reusing attention)
def param_init_gru_cond_reuse_att(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    if options['layer_normalisation']:
        sys.stderr.write('Warning: layer normalisation not implemented for gru_cond_reuse_att\n')

    return params


def gru_cond_layer_reuse_att(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=5)
    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=2)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, ctx_, h_, rec_dropout, ctx_dropout,
                    U, Wc, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):

        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        preact2 = tensor.dot(h1*rec_dropout[3], U_nl)+b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout[0], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout[1], Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2

    seqs = [mask, state_below_, state_belowx, context]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=init_state,
                                    non_sequences=[rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=True)
    return [rval]



def param_init_gru_local(options, params, prefix='gru_local',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att

    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    # local attention
    W_p = norm_weight(dim, dim)
    params[pp(prefix, 'W_p')] = W_p
    
    U_p = norm_weight(dim, 1)
    params[pp(prefix, 'U_p')] = U_p

    b_p = numpy.zeros((dim, )).astype(floatX)
    params[pp(prefix, 'b_p')] = b_p

    if options['layer_normalisation']:
        sys.stderr.write('Warning: layer normalisation not implemented for gru_local\n')

    return params

def gru_local_layer(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, 
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   profile=False, 
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    if pp(prefix, 'Wcx') in tparams:
        dim = tparams[pp(prefix, 'Wcx')].shape[1]
    else:
        dim = tparams[pp(prefix, 'Wcx_B')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=5)
    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'


    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]


    # projected x
    state_belowx = tensor.dot(state_below*below_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    winsize = 2 * options['pos_win'] + 1

    def _step_index(src_pos, one_sample):
        return one_sample[tensor.cast(src_pos, 'int64'): tensor.cast(src_pos + winsize, 'int64')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, cc_, rec_dropout, ctx_dropout,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl, W_p, U_p, b_p, Wc_att, b_att, indices_mask_, attention_mask_):

        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_  # n_samples * dim

        # local attention
        # predictive alignment
        msk_ = None
        if context_mask:
            msk_ = context_mask
        else:
            msk_ = tensor.alloc(1., cc_.shape[0], cc_.shape[1])   # (n_timestep, n_sample)

        sntlens = msk_.sum(0, keepdims=True)
        sntlens = sntlens.T   # n_sample * 1
        p_t = tensor.dot(h1, W_p) + b_p     # n_sample * dim
        p_t = tensor.tanh(p_t)       # n_sample * dim_predict(dim)
        p_t = tensor.dot(p_t, U_p)        # n_samples * 1
        p_t = sntlens * tensor.nnet.sigmoid(p_t)   # n_samples * 1
        src_positions = tensor.cast(tensor.floor(p_t), 'int64') # (n_samples, 1)
        src_positions = src_positions.reshape([src_positions.shape[0], ])

        p_t_steps = tensor.floor(p_t.reshape([p_t.shape[0], ]))
        indicesSub = indices_mask_[src_positions]       # n_samples, window 

        attn_mask = attention_mask_[src_positions, tensor.cast(sntlens.reshape([sntlens.shape[0], ]), 'int64')].T    # window, n_samples
        attn_mask_broad = tensor.tile(attn_mask[:, :, None], (1, 1, cc_.shape[2]))  # window, n_samples, dimctx

        cc_broad = concatenate([cc_, tensor.zeros([options['pos_win'], cc_.shape[1], cc_.shape[2]])], axis=0)
        cc_broad = concatenate([tensor.zeros([options['pos_win'], cc_.shape[1], cc_.shape[2]]), cc_broad], axis=0)  # (ntime+2win), nsample, dimctx
        ccshuffle = cc_broad.dimshuffle(1, 0, 2)      # n_sample, n_timestep+2win, dimctx

        # cc_result = ccshuffle[src_positions, ]
        cc_result, upd = theano.map(_step_index, 
                                sequences=[p_t_steps, ccshuffle])

        cc_result = cc_result.dimshuffle(1, 0, 2)  # window, n_sample, dimctx

        # p_t_gauss = tensor.tile(p_t_s, (winsize, 1)).reshape([winsize, msk_.shape[1]])   # window * n_sample
        gaussian_ = p_t.reshape([p_t.shape[0], ]) - indicesSub.T      # window * n_sample
        gaussian_ = (-0.5 * gaussian_ * gaussian_) / ((0.5 * options['pos_win']) ** 2)    # window * n_sample
        gaussian_ = tensor.exp(gaussian_) * attn_mask # window * n_sample

        ##########
        # attention
        # cc_slice is masked h_j
        b_att_broad = tensor.tile(b_att[None, None, :], (winsize, cc_.shape[1], 1)) * attn_mask_broad
        pctx_ = tensor.dot(cc_result*ctx_dropout[0], Wc_att) + b_att_broad   # window, n_sample, dimctx   U_ah_j
        pstate_ = tensor.dot(h1 * rec_dropout[2], W_comb_att)  # n_samples * 2dim       W_aS_{i-1}
        pstate_broad = tensor.tile(pstate_, (winsize, 1, 1)) * attn_mask_broad
        pctx__ = pctx_ + pstate_broad      # window * n_samples * 2dim + 1 * n_samples * 2dim
        pctx__ = tensor.tanh(pctx__) * attn_mask_broad    # window, n_samples, 2dim, masked
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att)   #+c_tt   # window * n_samples * 1
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])  # window * n_samples
        c_tt_broad = tensor.tile(c_tt[None, :], (attn_mask.shape[0], attn_mask.shape[1])) * attn_mask
        alpha += c_tt_broad
        alpha = tensor.exp(alpha) * attn_mask     # window, n_samples, masked
        alpha = alpha / alpha.sum(0, keepdims=True)   # (window, n_sample)
        alpha *= gaussian_ * attn_mask
        ctx_ = (cc_result * alpha[:, :, None]).sum(0)  # current context  (n_samples, dimctx)

        preact2 = tensor.dot(h1*rec_dropout[3], U_nl)+b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout[2], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout[3], Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T        # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[pp(prefix, 'U')],
               tparams[pp(prefix, 'Wc')],
               tparams[pp(prefix, 'W_comb_att')],
               tparams[pp(prefix, 'U_att')],
               tparams[pp(prefix, 'c_tt')],
               tparams[pp(prefix, 'Ux')],
               tparams[pp(prefix, 'Wcx')],
               tparams[pp(prefix, 'U_nl')],
               tparams[pp(prefix, 'Ux_nl')],
               tparams[pp(prefix, 'b_nl')],
               tparams[pp(prefix, 'bx_nl')], 
               tparams[pp(prefix, 'W_p')], 
               tparams[pp(prefix, 'U_p')], 
               tparams[pp(prefix, 'b_p')], 
               tparams[pp(prefix, 'Wc_att')], 
               tparams[pp(prefix, 'b_att')], 
               tparams['Indices_mask'], 
               tparams['Attention_mask']]


    if one_step:
        rval = _step(*(seqs + [init_state, None, None, context, rec_dropout, ctx_dropout] +
                      shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),    # ctx_: [n_samples * (dimctx=2dim)]
                                                  tensor.alloc(0., n_samples,
                                                               winsize)], 
                                    non_sequences=[context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval
