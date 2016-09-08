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
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[pp(prefix, 'W')]) +
        tparams[pp(prefix, 'b')])


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
            preact = tensor.dot(h_*rec_dropout[0], U)
            preact += x_

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_*rec_dropout[1], Ux)
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
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

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
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[pp(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   profile=False,
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

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context*ctx_dropout[0], tparams[pp(prefix, 'Wc_att')]) +\
        tparams[pp(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
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

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

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

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
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
                   tparams[pp(prefix, 'bx_nl')]]

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
                                    profile=profile,
                                    strict=True)
    return rval



