#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import numpy
import copy
import argparse

import os
import warnings
import sys
import time

import itertools

from subprocess import Popen

from collections import OrderedDict

profile = False

from data_iterator import TextIterator
from util import *
from theano_util import *
from alignment_util import *

from layers import *
from initializers import *
from optimizers import *
from metrics.scorer_provider import ScorerProvider

from domain_interpolation_data_iterator import DomainInterpolatorTextIterator

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

def init_attn_mask(win, maxlen):
    # return a maxlen * maxlen matrix
    # first dimension is middle point
    # second dimension is sentence length (middle point always less than sentence length, and sentence length == 0 is all 0)
    # last dimension is index
    mask = numpy.zeros((maxlen, maxlen, 2*win + 1), dtype='float32')
    for i in xrange(maxlen):
        start = i - win
        end = i + win
        for k in xrange(maxlen):
            idx = 0
            for j in xrange(start, end + 1):
                if j in xrange(k):
                    if i < k:
                        mask[i][k][idx] = 1
                idx += 1
    return mask

def init_index_mask(win, maxlen):
    # return a maxlen * window-size matrix
    winsize = 2 * win + 1
    mask = -1 * numpy.ones((maxlen, winsize), dtype='float32')
    for i in xrange(maxlen):
        start = i - win
        end = i + win
        idx = 0
        for j in xrange(start, end + 1):
            if j in xrange(maxlen):
                mask[i][idx] = j
            idx += 1
    return mask

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params = get_layer_param('embedding')(options, params, options['n_words_src'], options['dim_per_factor'], options['factors'], suffix='')
    if not options['tie_encoder_decoder_embeddings']:
        params = get_layer_param('embedding')(options, params, options['n_words'], options['dim_word'], suffix='_dec')

    if options['decoder'] == 'gru_local':
        params['Attention_mask'] = init_attn_mask(options['pos_win'], options['maxlen'])
        params['Indices_mask'] = init_index_mask(options['pos_win'], options['maxlen'])

    # encoder: bidirectional RNN
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer_param('ff')(options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer_param(options['decoder'])(options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    params = get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)


    params = get_layer_param('ff')(options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'],
                                weight_matrix = not options['tie_decoder_embeddings'])

    return params


# bidirectional RNN encoder: take input x (optionally with mask), and produce sequence of context vectors (ctx)
def build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=False):

    x = tensor.tensor3('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')

    # for the backward rnn, we just need to invert x
    xr = x[:,::-1]
    if x_mask is None:
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        if sampling:
            if options['model_version'] < 0.1:
                rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(retain_probability_source))
            else:
                rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(1.))
        else:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source, scaled)
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))

    # word embedding for forward rnn (source)
    emb = get_layer_constr('embedding')(tparams, x, suffix='', factors= options['factors'])
    if options['use_dropout']:
        emb *= source_dropout

    proj = get_layer_constr(options['encoder'])(tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            emb_dropout=emb_dropout,
                                            rec_dropout=rec_dropout,
                                            profile=profile)


    # word embedding for backward rnn (source)
    embr = get_layer_constr('embedding')(tparams, xr, suffix='', factors= options['factors'])
    if options['use_dropout']:
        if sampling:
            embr *= source_dropout
        else:
            # we drop out the same words in both directions
            embr *= source_dropout[::-1]

    projr = get_layer_constr(options['encoder'])(tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,
                                             profile=profile)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    return x, ctx


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
    y = tensor.matrix('y', dtype='int64')
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask, sampling=False)
    n_samples = x.shape[2]
    n_timesteps_trg = y.shape[0]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_target = 1-options['dropout_target']
        if options['model_version'] < 0.1:
            scaled = False
        else:
            scaled = True
        rec_dropout_d = shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctx_dropout_d = shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target, scaled)
        target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)

    # initial decoder state
    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
    emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
    if options['use_dropout']:
        emb *= target_dropout

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    if options['use_dropout']:
        proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)

    # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden, scaled)

    logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
    logit = get_layer_constr('ff')(tparams, logit, options,
                            prefix='ff_logit', activ='linear', W=logit_W)

    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):

    if options['use_dropout'] and options['model_version'] < 0.1:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=True)
    n_samples = x.shape[2]

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout'] and options['model_version'] < 0.1:
        ctx_mean *= retain_probability_hidden

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print >>sys.stderr, 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
    emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
    if options['use_dropout'] and options['model_version'] < 0.1:
        emb = emb * target_dropout
    emb = tensor.switch(y[:, None] < 0,
                        tensor.zeros((1, options['dim_word'])),
                        emb)


    # apply one step of conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model)
    dec_alphas = proj[2]

    if options['use_dropout'] and options['model_version'] < 0.1:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
        ctxs *= retain_probability_hidden
    else:
        next_state_up = next_state

    logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout'] and options['model_version'] < 0.1:
        logit *= retain_probability_hidden

    logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
    logit = get_layer_constr('ff')(tparams, logit, options,
                            prefix='ff_logit', activ='linear', W=logit_W)

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next...',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.append(dec_alphas)

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print >>sys.stderr, 'Done'

    return f_init, f_next


# minimum risk cost
# assumes cost is the negative sentence-level log probability
# and each sentence in the minibatch is a sample of the same source sentence
def mrt_cost(cost, y_mask, options):
    loss = tensor.vector('loss', dtype='float32')
    alpha = theano.shared(numpy.float32(options['mrt_alpha']))

    if options['mrt_ml_mix'] > 0:
        ml_cost = cost[0]

        # remove reference for MRT objective unless enabled
        if not options['mrt_reference']:
            cost = cost[1:]

    cost *= alpha

    #get normalized probability
    cost = tensor.nnet.softmax(-cost)[0]

    # risk: expected loss
    if options['mrt_ml_mix'] > 0 and not options['mrt_reference']:
        cost *= loss[1:]
    else:
        cost *= loss


    cost = cost.sum()

    if options['mrt_ml_mix'] > 0:
        #normalize ML by length (because MRT is length-invariant)
        ml_cost /= y_mask[:,0].sum(0)
        ml_cost *= options['mrt_ml_mix']
        cost += ml_cost

    return cost, loss


# build a sampler that produces samples in one theano function
def build_full_sampler(tparams, options, use_noise, trng, greedy=False):

    if options['use_dropout'] and options['model_version'] < 0.1:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))
        target_dropout = theano.shared(numpy.float32(1.))

    if greedy:
        x_mask = tensor.matrix('x_mask', dtype='float32')
        x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
    else:
        x_mask = None

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask, sampling=True)
    n_samples = x.shape[2]

    if x_mask:
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
    else:
        ctx_mean = ctx.mean(0)

    if options['use_dropout'] and options['model_version'] < 0.1:
        ctx_mean *= retain_probability_hidden

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    if greedy:
        init_w = tensor.alloc(numpy.int64(-1), n_samples)
    else:
        k = tensor.iscalar("k")
        k.tag.test_value = 12
        init_w = tensor.alloc(numpy.int64(-1), k*n_samples)

        ctx = tensor.tile(ctx, [k, 1])

        init_state = tensor.tile(init_state, [k, 1])

    # projected context
    assert ctx.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(ctx*ctx_dropout_d[0], tparams[pp('decoder', 'Wc_att')]) +\
        tparams[pp('decoder', 'b_att')]

    def decoder_step(y, init_state, ctx, pctx_, target_dropout, emb_dropout, rec_dropout, ctx_dropout, *shared_vars):

        # if it's the first word, emb should be all zero and it is indicated by -1
        decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
        emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
        emb = tensor.switch(y[:, None] < 0,
                            tensor.zeros((1, options['dim_word'])),
                            emb)
        emb *= target_dropout

        # apply one step of conditional gru with attention
        proj = get_layer_constr('gru_cond')(tparams, emb, options,
                                                prefix='decoder',
                                                mask=None,
                                                context=ctx,
                                                context_mask=x_mask,
                                                pctx_=pctx_,
                                                one_step=True,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout,
                                                ctx_dropout=ctx_dropout,
                                                rec_dropout=rec_dropout,
                                                shared_vars=shared_vars,
                                                profile=profile)
        # get the next hidden state
        next_state = proj[0]

        # get the weighted averages of context for this target word y
        ctxs = proj[1]

        # alignment matrix (attention model)
        dec_alphas = proj[2]

        if options['use_dropout'] and options['model_version'] < 0.1:
            next_state_up = next_state * retain_probability_hidden
            emb *= retain_probability_emb
            ctxs *= retain_probability_hidden
        else:
            next_state_up = next_state

        logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                    prefix='ff_logit_ctx', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

        if options['use_dropout'] and options['model_version'] < 0.1:
            logit *= retain_probability_hidden

        logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
        logit = get_layer_constr('ff')(tparams, logit, options,
                                prefix='ff_logit', activ='linear', W=logit_W)

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)

        if greedy:
            next_sample = next_probs.argmax(1)
        else:
            # sample from softmax distribution to get the sample
            next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # do not produce words after EOS
        next_sample = tensor.switch(
                      tensor.eq(y,0),
                      0,
                      next_sample)

        return [next_sample, next_state, next_probs[:, next_sample].diagonal()], \
               theano.scan_module.until(tensor.all(tensor.eq(next_sample, 0))) # stop when all outputs are 0 (EOS)

    # symbolic loop for sequence generation
    shared_vars = [tparams[pp('decoder', 'U')],
                   tparams[pp('decoder', 'Wc')],
                   tparams[pp('decoder', 'W_comb_att')],
                   tparams[pp('decoder', 'U_att')],
                   tparams[pp('decoder', 'c_tt')],
                   tparams[pp('decoder', 'Ux')],
                   tparams[pp('decoder', 'Wcx')],
                   tparams[pp('decoder', 'U_nl')],
                   tparams[pp('decoder', 'Ux_nl')],
                   tparams[pp('decoder', 'b_nl')],
                   tparams[pp('decoder', 'bx_nl')]]


    n_steps = tensor.iscalar("n_steps")
    n_steps.tag.test_value = 50

    (sample, state, probs), updates = theano.scan(decoder_step,
                        outputs_info=[init_w, init_state, None],
                        non_sequences=[ctx, pctx_, target_dropout, emb_dropout_d, rec_dropout_d, ctx_dropout_d]+shared_vars,
                        n_steps=n_steps)

    print >>sys.stderr, 'Building f_sample...',
    if greedy:
        inps = [x, x_mask, n_steps]
    else:
        inps = [x, k, n_steps]
    outs = [sample, probs]
    f_sample = theano.function(inps, outs, name='f_sample', updates=updates, profile=profile)
    print >>sys.stderr, 'Done'

    return f_sample



# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False,
               return_hyp_graph=False):

    # k is the beam size we have
    if k > 1 and argmax:
        assert not stochastic, \
            'Beam search does not support stochastic sampling with argmax'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
    hyp_graph = None
    if stochastic:
        if argmax:
            sample_score = 0
        live_k=k
    else:
        live_k = 1

    if return_hyp_graph:
        from hypgraph import HypGraph
        hyp_graph = HypGraph()

    dead_k = 0

    hyp_samples=[ [] for i in xrange(live_k) ]
    word_probs=[ [] for i in xrange(live_k) ]
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    dec_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)
        next_state[i] = numpy.tile( ret[0] , (live_k,1))
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((live_k,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])
            inps = [next_w, ctx, next_state[i]]
            ret = f_next[i](*inps)
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:
            #batches are not supported with argmax: output data structure is different
            if argmax:
                nw = sum(next_p)[0].argmax()
                sample.append(nw)
                sample_score += numpy.log(next_p[0][0, nw])
                if nw == 0:
                    break
            else:
                #FIXME: sampling is currently performed according to the last model only
                nws = next_w_tmp
                cand_scores = numpy.array(hyp_scores)[:, None] - numpy.log(next_p[-1])
                probs = next_p[-1]

                for idx,nw in enumerate(nws):
                    hyp_samples[idx].append(nw)


                hyp_states=[]
                for ti in xrange(live_k):
                    hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                    hyp_scores[ti]=cand_scores[ti][nws[ti]]
                    word_probs[ti].append(probs[ti][nws[ti]])

                new_hyp_states=[]
                new_hyp_samples=[]
                new_hyp_scores=[]
                new_word_probs=[]
                for hyp_sample,hyp_state, hyp_score, hyp_word_prob in zip(hyp_samples,hyp_states,hyp_scores, word_probs):
                    if hyp_sample[-1]  > 0:
                        new_hyp_samples.append(copy.copy(hyp_sample))
                        new_hyp_states.append(copy.copy(hyp_state))
                        new_hyp_scores.append(hyp_score)
                        new_word_probs.append(hyp_word_prob)
                    else:
                        sample.append(copy.copy(hyp_sample))
                        sample_score.append(hyp_score)
                        sample_word_probs.append(hyp_word_prob)

                hyp_samples=new_hyp_samples
                hyp_states=new_hyp_states
                hyp_scores=new_hyp_scores
                word_probs=new_word_probs

                live_k=len(hyp_samples)
                if live_k < 1:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = [numpy.array(state) for state in zip(*hyp_states)]
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            #averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if return_hyp_graph:
                    word, history = new_hyp_samples[idx][-1], new_hyp_samples[idx][:-1]
                    score = new_hyp_scores[idx]
                    word_prob = new_word_probs[idx][-1]
                    hyp_graph.add(word, history, word_prob=word_prob, cost=score)
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(copy.copy(new_hyp_samples[idx]))
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(copy.copy(new_hyp_samples[idx]))
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(copy.copy(new_hyp_states[idx]))
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    # dump every remaining one
    if not argmax and live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])
            sample_word_probs.append(word_probs[idx])
            if return_alignment:
                alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment, hyp_graph


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        #ensure consistency in number of factors
        if len(x[0][0]) != options['factors']:
            sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        ### in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs), alignments_json


def train(dim_word=512,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          factors=1, # input factors
          dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=1000,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0., # L2 regularization penalty towards original weights
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.0001,  # learning rate
          n_words_src=None,  # source vocabulary size
          n_words=None,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          pos_win=10, # half window size of local attenton
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=10000,
          saveFreq=30000,   # save the parameters after every saveFreq updates
          sampleFreq=10000,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
          reload_=False,
          overwrite=False,
          external_validation_script=None,
          shuffle_each_epoch=True,
          finetune=False,
          finetune_only_last=False,
          sort_by_length=True,
          use_domain_interpolation=False,
          domain_interpolation_min=0.1,
          domain_interpolation_inc=0.1,
          domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'],
          maxibatch_size=20, #How many minibatches to load at one time
          objective="CE", #CE: cross-entropy; MRT: minimum risk training (see https://www.aclweb.org/anthology/P/P16/P16-1159.pdf)
          mrt_alpha=0.005,
          mrt_samples=100,
          mrt_reference=False,
          mrt_loss="SENTENCEBLEU n=4", # loss function for minimum risk training
          mrt_ml_mix=0, # interpolate mrt loss with ML loss
          model_version=0.1, #store version used for training for compatibility
          prior_model=None, # Prior model file, used for MAP
          tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
          tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
    ):

    # Model options
    model_options = locals().copy()


    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(dictionaries) == factors + 1) # one dictionary per source factor + 1 for target factor
    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    assert(prior_model != None and (os.path.exists(prior_model)) or (map_decay_c==0.0)) # MAP training requires a prior model file

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    if n_words_src is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if n_words is None:
        n_words = len(worddicts[1])
        model_options['n_words'] = n_words

    if tie_encoder_decoder_embeddings:
        assert (n_words_src == n_words), "When tying encoder and decoder embeddings, source and target vocabulary size must the same"
        if worddicts[0] != worddicts[1]:
            warn("Encoder-decoder embedding tying is enabled with different source and target dictionaries. This is usually a configuration error")

    if model_options['objective'] == 'MRT':
        # in CE mode parameters are updated once per batch; in MRT mode parameters are updated once
        # per pair of train sentences (== per batch of samples), so we set batch_size to 1 to make
        # model saving, validation, etc trigger after the same number of updates as before
        print 'Running in MRT mode, minibatch size set to 1 sentence'
        batch_size = 1

    print 'Loading data'
    domain_interpolation_cur = None
    if use_domain_interpolation:
        print 'Using domain interpolation with initial ratio %s, increase rate %s' % (domain_interpolation_min, domain_interpolation_inc)
        domain_interpolation_cur = domain_interpolation_min
        train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
                         dictionaries[:-1], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         skip_empty=True,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         indomain_source=domain_interpolation_indomain_datasets[0],
                         indomain_target=domain_interpolation_indomain_datasets[1],
                         interpolation_rate=domain_interpolation_cur,
                         maxibatch_size=maxibatch_size)
    else:
        train = TextIterator(datasets[0], datasets[1],
                         dictionaries[:-1], dictionaries[-1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         skip_empty=True,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets[0], valid_datasets[1],
                            dictionaries[:-1], dictionaries[-1],
                            n_words_source=n_words_src, n_words_target=n_words,
                            batch_size=valid_batch_size,
                            maxlen=maxlen)
    else:
        valid = None

    comp_start = time.time()

    print 'Building model'
    params = init_params(model_options)

    optimizer_params = {}
    # prepare parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)
        print 'Reloading optimizer parameters'
        optimizer_params = load_optimizer_params(saveto, optimizer)
    elif prior_model:
        print 'Initializing model parameters from prior'
        params = load_params(prior_model, params)

    # load prior model if specified
    if prior_model:
        print 'Loading prior model parameters'
        params = load_params(prior_model, params, with_prefix='prior_')

    tparams = init_theano_params(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)

    inps = [x, x_mask, y, y_mask]

    if validFreq or sampleFreq:
        print 'Building sampler'
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)
    if model_options['objective'] == 'MRT':
        print 'Building MRT sampler'
        f_sampler = build_full_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    if model_options['objective'] == 'CE':
        cost = cost.mean()
    elif model_options['objective'] == 'MRT':
        #MRT objective function
        cost, loss = mrt_cost(cost, y_mask, model_options)
        inps += [loss]
    else:
        sys.stderr.write('Error: objective must be one of ["CE", "MRT"]\n')
        sys.exit(1)

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.startswith('prior_'):
                continue
            init_value = tparams['prior_' + kk]
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay

    updated_params = OrderedDict(tparams)

    # don't update prior model parameters
    if prior_model:
        updated_params = OrderedDict([(key,value) for (key,value) in updated_params.iteritems() if not key.startswith('prior_')])

    # allow finetuning with fixed embeddings
    if finetune:
        updated_params = OrderedDict([(key,value) for (key,value) in updated_params.iteritems() if not key.startswith('Wemb')])

    # allow finetuning of only last layer (becomes a linear model training problem)
    if finetune_only_last:
        updated_params = OrderedDict([(key,value) for (key,value) in updated_params.iteritems() if key in ['ff_logit_W', 'ff_logit_b']])

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    print 'Building optimizers...',
    f_grad_shared, f_update, optimizer_tparams = eval(optimizer)(lr, updated_params,
                                                                 grads, inps, cost,
                                                                 profile=profile,
                                                                 optimizer_params=optimizer_params)
    print 'Done'

    print 'Total compilation time: {0:.1f}s'.format(time.time() - comp_start)

    print 'Optimization'

    best_p = None
    best_opt_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    #save model options
    json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    valid_err = None

    cost_sum = 0
    cost_batches = 0
    last_disp_samples = 0
    last_words = 0
    ud_start = time.time()
    p_validation = None
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            uidx += 1
            use_noise.set_value(1.)

            #ensure consistency in number of factors
            if len(x) and len(x[0]) and len(x[0][0]) != factors:
                sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(factors, len(x[0][0])))
                sys.exit(1)

            if model_options['objective'] == 'CE':

                xlen = len(x)
                n_samples += xlen
                x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                    n_words_src=n_words_src,
                                                    n_words=n_words)

                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                cost_batches += 1
                last_disp_samples += xlen
                last_words += (numpy.sum(x_mask) + numpy.sum(y_mask))/2.0

                # compute cost, grads and copy grads to shared variables
                cost = f_grad_shared(x, x_mask, y, y_mask)
                cost_sum += cost

                # do the update on parameters
                f_update(lrate)

            elif model_options['objective'] == 'MRT':
                assert maxlen is not None and maxlen > 0

                xy_pairs = [(x_i, y_i) for (x_i, y_i) in zip(x, y) if len(x_i) < maxlen and len(y_i) < maxlen]
                if not xy_pairs:
                    uidx -= 1
                    continue

                for x_s, y_s in xy_pairs:

                    # create k samples
                    use_noise.set_value(0.)
                    samples, _ = f_sampler([x_s], model_options['mrt_samples'], maxlen)
                    use_noise.set_value(1.)

                    samples = [numpy.trim_zeros(item) for item in zip(*samples)]

                    # remove duplicate samples
                    samples.sort()
                    samples = [s for s, _ in itertools.groupby(samples)]

                    # add gold translation [always in first position]
                    if model_options['mrt_reference'] or model_options['mrt_ml_mix']:
                        samples = [y_s] + [s for s in samples if s != y_s]

                    # create mini-batch with masking
                    x, x_mask, y, y_mask = prepare_data([x_s for _ in xrange(len(samples))], samples,
                                                                    maxlen=None, n_words_src=n_words_src,
                                                                    n_words=n_words)

                    # map integers to words (for character-level metrics)
                    samples = [seqs2words(sample, worddicts_r[-1]) for sample in samples]
                    y_s = seqs2words(y_s, worddicts_r[-1])

                    #scorers expect tokenized hypotheses/references
                    y_s = y_s.split(" ")
                    samples = [sample.split(" ") for sample in samples]

                    # get negative smoothed BLEU for samples
                    scorer = ScorerProvider().get(model_options['mrt_loss'])
                    scorer.set_reference(y_s)
                    loss = 1-numpy.array(scorer.score_matrix(samples), dtype='float32')

                    # compute cost, grads and copy grads to shared variables
                    cost = f_grad_shared(x, x_mask, y, y_mask, loss)
                    # do the update on parameters
                    f_update(lrate)

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                sps = last_disp_samples / float(ud)
                wps = last_words / float(ud)
                cost_avg = cost_sum / float(cost_batches)
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost_avg, 'UD ', ud, "{0:.2f} sents/s".format(sps), "{0:.2f} words/s".format(wps)
                ud_start = time.time()
                cost_batches = 0
                last_disp_samples = 0
                last_words = 0
                cost_sum = 0

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                    optimizer_params = best_opt_p
                else:
                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

                both_params = dict(params, **optimizer_params)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **both_params)
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)

                    both_params = dict(unzip_from_theano(tparams, excluding_prefix='prior_'), **unzip_from_theano(optimizer_tparams, excluding_prefix='prior_'))
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **both_params)
                    print 'Done'


            # generate some samples with the model and display them
            if sampleFreq and numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[2])):
                    stochastic = True
                    x_current = x[:, :, jj][:, :, None]

                    # remove padding
                    x_current = x_current[:,:x_mask.astype('int64')[:, jj].sum(),:]

                    sample, score, sample_word_probs, alignment, hyp_graph = gen_sample([f_init], [f_next],
                                               x_current,
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False,
                                               return_hyp_graph=False)
                    print 'Source ', jj, ': ',
                    for pos in range(x.shape[1]):
                        if x[0, pos, jj] == 0:
                            break
                        for factor in range(factors):
                            vv = x[factor, pos, jj]
                            if vv in worddicts_r[factor]:
                                sys.stdout.write(worddicts_r[factor][vv])
                            else:
                                sys.stdout.write('UNK')
                            if factor+1 < factors:
                                sys.stdout.write('|')
                            else:
                                sys.stdout.write(' ')
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample[0]
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip_from_theano(tparams, excluding_prefix='prior_')
                    best_opt_p = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if use_domain_interpolation and (domain_interpolation_cur < 1.0):
                            domain_interpolation_cur = min(domain_interpolation_cur + domain_interpolation_inc, 1.0)
                            print 'No progress on the validation set, increasing domain interpolation rate to %s and resuming from best params' % domain_interpolation_cur
                            train.adjust_domain_interpolation_rate(domain_interpolation_cur)
                            if best_p is not None:
                                zip_to_theano(best_p, tparams)
                                zip_to_theano(best_opt_p, optimizer_tparams)
                            bad_counter = 0
                        else:
                            print 'Early Stop!'
                            estop = True
                            break

                print 'Valid ', valid_err

                if external_validation_script:
                    print "Calling external validation script"
                    if p_validation is not None and p_validation.poll() is None:
                        print "Waiting for previous validation run to finish"
                        print "If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes"
                        valid_wait_start = time.time()
                        p_validation.wait()
                        print "Waited for {0:.1f} seconds".format(time.time()-valid_wait_start)
                    print 'Saving  model...',
                    params = unzip_from_theano(tparams, excluding_prefix='prior_')
                    optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')
                    both_params = dict(params, **optimizer_params)
                    numpy.savez(saveto +'.dev', history_errs=history_errs, uidx=uidx, **both_params)
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p_validation = Popen([external_validation_script])

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zip_to_theano(best_p, tparams)
        zip_to_theano(best_opt_p, optimizer_tparams)

    if valid:
        use_noise.set_value(0.)
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
        valid_err = valid_errs.mean()

        print 'Valid ', valid_err

    if best_p is not None:
        params = copy.copy(best_p)
        optimizer_params = copy.copy(best_opt_p)

    else:
        params = unzip_from_theano(tparams, excluding_prefix='prior_')
        optimizer_params = unzip_from_theano(optimizer_tparams, excluding_prefix='prior_')

    both_params = dict(params, **optimizer_params)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **both_params)

    return valid_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs=2,
                         help="parallel training corpus (source and target)")
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (one per source factor, plus target vocabulary)")
    data.add_argument('--model', type=str, default='model.npz', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--reload', action='store_true',  dest='reload_',
                         help="load existing model (if '--model' points to existing model)")
    data.add_argument('--overwrite', action='store_true',
                         help="write all models to same file")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dim', type=int, default=1000, metavar='INT',
                         help="hidden layer size (default: %(default)s)")
    network.add_argument('--n_words_src', type=int, default=None, metavar='INT',
                         help="source vocabulary size (default: %(default)s)")
    network.add_argument('--n_words', type=int, default=None, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")

    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")
    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--use_dropout', action="store_true",
                         help="use dropout layer (default: %(default)s)")
    network.add_argument('--dropout_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for input embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_source', type=float, default=0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")
    network.add_argument('--tie_encoder_decoder_embeddings', action="store_true", dest="tie_encoder_decoder_embeddings",
                         help="tie the input embeddings of the encoder and the decoder (first factor only). Source and target vocabulary size must the same")
    network.add_argument('--tie_decoder_embeddings', action="store_true", dest="tie_decoder_embeddings",
                         help="tie the input embeddings of the decoder with the softmax output embeddings")
    #network.add_argument('--encoder', type=str, default='gru',
                         #choices=['gru'],
                         #help='encoder recurrent layer')
    network.add_argument('--decoder', type=str, default='gru_cond',
                         choices=['gru_cond', 'gru_local'],
                         help='decoder recurrent layer')

    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--pos_win', type=int, default=10, metavar='INT',
                         help="half window size of local attention (default: %(default)s)")
    training.add_argument('--optimizer', type=str, default="adam",
                         choices=['adam', 'adadelta', 'rmsprop', 'sgd'],
                         help="optimizer (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=10000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--decay_c', type=float, default=0, metavar='FLOAT',
                         help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0, metavar='FLOAT',
                         help="L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--alpha_c', type=float, default=0, metavar='FLOAT',
                         help="alignment regularization (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument('--objective', choices=['CE', 'MRT'], default='CE',
                         help='training objective. CE: cross-entropy minimization (default); MRT: Minimum Risk Training (https://www.aclweb.org/anthology/P/P16/P16-1159.pdf)')

    finetune = training.add_mutually_exclusive_group()
    finetune.add_argument('--finetune', action="store_true",
                        help="train with fixed embedding layer")
    finetune.add_argument('--finetune_only_last', action="store_true",
                        help="train with all layers except output layer fixed")

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs=2,
                         help="parallel validation corpus (source and target) (default: %(default)s)")
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validation minibatch size (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--external_validation_script', type=str, default=None, metavar='PATH',
                         help="location of validation script (to run your favorite metric for validation) (default: %(default)s)")

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")

    mrt = parser.add_argument_group('minimum risk training parameters')
    mrt.add_argument('--mrt_alpha', type=float, default=0.005, metavar='FLOAT',
                         help="MRT alpha (default: %(default)s)")
    mrt.add_argument('--mrt_samples', type=int, default=100, metavar='INT',
                         help="samples per source sentence (default: %(default)s)")
    mrt.add_argument('--mrt_loss', type=str, default='SENTENCEBLEU n=4', metavar='STR',
                         help='loss used in MRT (default: %(default)s)')
    mrt.add_argument('--mrt_reference', action="store_true",
                         help='add reference to MRT samples.')
    mrt.add_argument('--mrt_ml_mix', type=float, default=0, metavar='FLOAT',
                     help="Mix in ML objective in MRT training with this scaling factor (default: %(default)s)")

    args = parser.parse_args()

    train(**vars(args))
