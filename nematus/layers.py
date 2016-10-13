'''
Layer definitions
'''

import sys
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

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lrd_ff': ('param_init_lrd_fflayer', 'lrd_fflayer'),
          'multi_lrd_ff': ('param_init_multi_lrd_fflayer', 'multi_lrd_fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer_param(name):
    param_fn, constr_fn = layers[name]
    return eval(param_fn)

def get_layer_constr(name):
    param_fn, constr_fn = layers[name]
    return eval(constr_fn)

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value):
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1,
                                     dtype='float32'),
        theano.shared(numpy.float32(value)))
    return proj

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, with_bias=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if with_bias:
        params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='ff',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    pre_act = tensor.dot(state_below, tparams[pp(prefix, 'W')])
    if pp(prefix, 'b') in tparams:
        pre_act += tparams[pp(prefix, 'b')]
    return eval(activ)(pre_act)

# low-rank plus diagonal feedforward layer. Input dimension and output dimension must multiples of each other unless rank is 'full'
def param_init_lrd_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True, with_bias=True, rank='full', plus_diagonal=True, init_proj_matrix=True):
    if rank == 'full':
        return param_init_fflayer(options, params, prefix, nin, nout, ortho, with_bias)

    max_dim, min_dim = max(nin, nout), min(nin, nout)
    assert (max_dim % min_dim) == 0
    dim_ratio = max_dim / min_dim
    
    if init_proj_matrix:
        params[pp(prefix, 'W_proj')] = norm_weight(nin, rank, scale=0.01)
    params[pp(prefix, 'W_expand')] = norm_weight(rank, nout, scale=0.01)
    if plus_diagonal:
        params[pp(prefix, 'd')] = numpy.ones((max_dim,)).astype('float32')
    if with_bias:
        params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def lrd_fflayer(tparams, state_below, options, prefix='ff',
            activ='lambda x: tensor.tanh(x)', proj_matrix_name=None, **kwargs):
    if pp(prefix, 'W') in tparams:    # Full rank
        return fflayer(tparams, state_below, options, prefix, activ, **kwargs)

    if proj_matrix_name == None:
        proj_matrix_name = pp(prefix, 'W_proj')
    nin  = tparams[proj_matrix_name].get_value().shape[0]
    nout = tparams[pp(prefix, 'W_expand')].get_value().shape[1]
    plus_diagonal = pp(prefix, 'd') in tparams
    max_dim, min_dim = max(nin, nout), min(nin, nout)
    assert (max_dim % min_dim) == 0
    dim_ratio = max_dim / min_dim
    
    pre_act = tensor.dot(state_below, tparams[proj_matrix_name]).dot(tparams[pp(prefix, 'W_expand')])
    if plus_diagonal:
        x = state_below
        if nin < nout:
            # input dimension is smaller than output dimension, replicate over input dimension
            x = tensor.repeat(x, dim_ratio, axis=x.ndim-1)
            pre_act += x * tparams[pp(prefix, 'd')]
        elif nin > nout:
            # input dimension is larger than output dimension, reduce
            shape = list(x.shape)
            shape[-1] = nout
            shape = shape + [dim_ratio]
            dd = x * tparams[pp(prefix, 'd')]
            dd = dd.reshape(shape)
            dd = dd.sum(axis = len(shape) - 1)
            pre_act += dd
        else:
            pre_act += x * tparams[pp(prefix, 'd')]
    if pp(prefix, 'b') in tparams:
        pre_act += tparams[pp(prefix, 'b')]
    
    activ_fn = eval(activ)
    return activ_fn(pre_act)

# Multi-output fullrank or low-rank plus diagonal feedforward layer.
# Allow projection matrix sharing in low-rank and low-rank plus diagonal mode.
# Input dimension and output dimension must multiples of each other unless rank is 'full'
def param_init_multi_lrd_fflayer(options, params, prefix='multi_ff', n_output_groups=None, nin=None, nout=None, ortho=True, with_bias=True, rank='full', plus_diagonal=True, share_proj_matrix=False):
    if rank == 'full':
        for i in xrange(n_output_groups):
            params = param_init_fflayer(options, params, prefix+('_%s' % i), nin, nout, ortho, with_bias)
    else:
        for i in xrange(n_output_groups):
            init_proj_matrix = (i == 0) or (share_proj_matrix==False)
            params = param_init_lrd_fflayer(options, params, prefix+('_%s' % i), nin, nout, ortho, with_bias, rank, plus_diagonal, init_proj_matrix)
    return params

def multi_lrd_fflayer(tparams, state_below, options, prefix='multi_ff', n_output_groups=None, 
            activ='lambda x: tensor.tanh(x)', **kwargs):
    if pp(prefix+'_0', 'W') in tparams:    # Full rank
        rv = []
        for i in xrange(n_output_groups):
             y_i = fflayer(tparams, state_below, options, prefix+('_%s' % i), activ, **kwargs)
             rv.append(y_i)
    else:
        share_proj_matrix = not (pp(prefix+'_1', 'W_proj') in tparams)
        proj_matrix_name = pp(prefix+'_0', 'W_proj') if share_proj_matrix else None
        rv = []
        for i in xrange(n_output_groups):
            y_i = lrd_fflayer(tparams, state_below, options, prefix+('_%s' % i), activ, proj_matrix_name, **kwargs)
            rv.append(y_i)
    return rv
    
# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, rank='full', share_proj_matrix=False, plus_diagonal=True):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    if rank == 'full':
        # recurrent transformation weights for gates
        U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
        params[pp(prefix, 'U')] = U

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux')] = Ux
    else:
        if share_proj_matrix:
            U_proj = norm_weight(dim, rank)
            params[pp(prefix, 'U_proj')] = U_proj
        else:
            U_proj_u = norm_weight(dim, rank)
            U_proj_r = norm_weight(dim, rank)
            U_proj_x = norm_weight(dim, rank)
            params[pp(prefix, 'U_proj_u')] = U_proj_u
            params[pp(prefix, 'U_proj_r')] = U_proj_r
            params[pp(prefix, 'U_proj_x')] = U_proj_x
        U_expand_u = norm_weight(rank, dim)
        U_expand_r = norm_weight(rank, dim)
        U_expand_x = norm_weight(rank, dim)
        params[pp(prefix, 'U_expand_u')] = U_expand_u
        params[pp(prefix, 'U_expand_r')] = U_expand_r
        params[pp(prefix, 'U_expand_x')] = U_expand_x
        if plus_diagonal:
            U_diag_u = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            U_diag_r = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            U_diag_x = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            params[pp(prefix, 'U_diag_u')] = U_diag_u
            params[pp(prefix, 'U_diag_r')] = U_diag_r
            params[pp(prefix, 'U_diag_x')] = U_diag_x

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              emb_dropout=None,
              rec_dropout=None,
              profile=False,
              rank='full',
              share_proj_matrix=False,
              plus_diagonal=True,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[pp(prefix, 'Wx')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'W')]) + \
        tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'Wx')]) + \
        tparams[pp(prefix, 'bx')]
            
    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_):
        if rank == 'full':
            preact = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U')])
            preact += x_

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_*rec_dropout[1], tparams[pp(prefix, 'Ux')])
            preactx = preactx * r
            preactx = preactx + xx_
        else:
            if share_proj_matrix:
                proj = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U_proj')])
                proj_u, proj_r, proj_x = proj, proj, proj
            else:
                proj_u = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U_proj_u')])
                proj_r = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U_proj_r')])
                proj_x = tensor.dot(h_*rec_dropout[1], tparams[pp(prefix, 'U_proj_x')])
            preact_u = tensor.dot(proj_u, tparams[pp(prefix, 'U_expand_u')]) + _slice(x_, 0, dim)
            preact_r = tensor.dot(proj_r, tparams[pp(prefix, 'U_expand_r')]) + _slice(x_, 1, dim)
            if plus_diagonal:
                 preact_u += h_*rec_dropout[0] * tparams[pp(prefix, 'U_diag_u')]
                 preact_r += h_*rec_dropout[0] * tparams[pp(prefix, 'U_diag_r')]
            u = tensor.nnet.sigmoid(preact_u)
            r = tensor.nnet.sigmoid(preact_r)
            pre_preactx = tensor.dot(proj_x, tparams[pp(prefix, 'U_expand_x')])
            if plus_diagonal:
                pre_preactx += h_*rec_dropout[1] * tparams[pp(prefix, 'U_diag_x')]
            preactx = pre_preactx * r + xx_
            

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        rank='full', share_proj_matrix=False, plus_diagonal=True):
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

    # gates inputs
    params = get_layer_param('multi_lrd_ff')(options, params, prefix=prefix+'_embeddings_to_state', n_output_groups=3,
                                nin=nin, nout=dim, rank='full')

    params = get_layer_param('multi_lrd_ff')(options, params, prefix=prefix+'_context_to_state', n_output_groups=3,
                                nin=dimctx, nout=dim, rank=options['context_rank'], plus_diagonal=options['context_plus_diagonal'], share_proj_matrix=options['context_share_proj_matrix'])

    # gates recurrence
    params = get_layer_param('multi_lrd_ff')(options, params, prefix=prefix+'_state_to_state_layer0', n_output_groups=3,
                                nin=dim, nout=dim, rank=rank, plus_diagonal=plus_diagonal, share_proj_matrix=share_proj_matrix, with_bias=False)
    
    params = get_layer_param('multi_lrd_ff')(options, params, prefix=prefix+'_state_to_state_layer1', n_output_groups=3,
                                nin=dim, nout=dim, rank=rank, plus_diagonal=plus_diagonal, share_proj_matrix=share_proj_matrix, with_bias=False)

    # attention (MLP with one tanh hidden layer)

    params = get_layer_param('lrd_ff')(options, params, prefix=prefix+'_state_to_attention_hidden',
                                nin=dim, nout=dimctx, rank=options['context_rank'], plus_diagonal=options['context_plus_diagonal'], with_bias=False)

    params = get_layer_param('lrd_ff')(options, params, prefix=prefix+'_context_to_attention_hidden',
                                nin=dimctx, nout=dimctx, rank=options['context_rank'], plus_diagonal=options['context_plus_diagonal'])

    params = get_layer_param('ff')(options, params, prefix=prefix+'_attention_hidden_to_attention_final',
                                nin=dimctx, nout=1)

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   profile=False,
                   rank='full', share_proj_matrix=False, plus_diagonal=True,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix+'_embeddings_to_state_0', 'W')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    init_state.name='init_state'

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    #pctx_ = tensor.dot(context*ctx_dropout[0], tparams[pp(prefix, 'Wc_att')]) +\
    #    tparams[pp(prefix, 'b_att')]

    pctx_ = get_layer_constr('lrd_ff')(tparams, context*ctx_dropout[0], options, 
                                    prefix=prefix+'_context_to_attention_hidden', activ='linear')

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    in_x, in_r, in_u = get_layer_constr('multi_lrd_ff')(tparams, state_below*emb_dropout[0], options, n_output_groups=3,
                                    prefix=prefix+'_embeddings_to_state', activ='linear')

    def _step_slice(m_, in_x_, in_r_, in_u_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout):
        # first layer
        rec0_x, rec0_r, rec0_u = get_layer_constr('multi_lrd_ff')(tparams, h_*rec_dropout[0], options, n_output_groups=3,
                                    prefix=prefix+'_state_to_state_layer0', activ='linear')
        rec0_x.name = 'rec0_x'
        rec0_r.name = 'rec0_r'
        rec0_u.name = 'rec0_u'
        in_x_.name = 'in_x_'
        in_r_.name = 'in_r_'
        in_u_.name = 'in_u_'

        r0 = tensor.nnet.sigmoid(rec0_r + in_r_)
        u0 = tensor.nnet.sigmoid(rec0_u + in_u_)
        x0 = tensor.tanh(rec0_x * r0 + in_x_)

        h1 = u0 * h_ + (1. - u0) * x0
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_
        h1.name='h1'

        # attention
        pstate_ = get_layer_constr('lrd_ff')(tparams, h1*rec_dropout[2], options,
                                    prefix=prefix+'_state_to_attention_hidden', activ='linear')
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = tensor.tanh(pctx__)
        alpha = get_layer_constr('ff')(tparams, pctx__*ctx_dropout[1], options,
                                    prefix=prefix+'_attention_hidden_to_attention_final', activ='linear')
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        alpha.name='alpha_'
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context
        ctx_.name='ctx_'

        # second layer
        ctx_x, ctx_r, ctx_u    = get_layer_constr('multi_lrd_ff')(tparams, ctx_*ctx_dropout[2], options, n_output_groups=3,
                                    prefix=prefix+'_context_to_state', activ='linear')
        rec1_x, rec1_r, rec1_u = get_layer_constr('multi_lrd_ff')(tparams, h_*rec_dropout[4], options, n_output_groups=3,
                                    prefix=prefix+'_state_to_state_layer1', activ='linear')
        rec1_x.name = 'rec1_x'
        rec1_r.name = 'rec1_r'
        rec1_u.name = 'rec1_u'
        ctx_x.name = 'ctx_x'
        ctx_r.name = 'ctx_r'
        ctx_u.name = 'ctx_u'

        r1 = tensor.nnet.sigmoid(rec1_r + ctx_r)
        u1 = tensor.nnet.sigmoid(rec1_u + ctx_u)
        x1 = tensor.tanh(rec1_x * r1 + ctx_x)

        h2 = u1 * h1 + (1. - u1) * x1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1
        h2.name='h2'

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, in_x, in_r, in_u]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice


    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout]))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout],
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=False)
    return rval



