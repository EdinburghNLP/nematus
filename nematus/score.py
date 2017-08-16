#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score, and optionally alignment for each pair.
"""

import sys
import argparse
import tempfile
import logging

import numpy

from data_iterator import TextIterator
from util import load_config
from alignment_util import combine_source_target_text_1to1
from compat import fill_options

from theano_util import (floatX, numpy_floatX, load_params, init_theano_params)
from nmt import (pred_probs, build_model, prepare_data)
from settings import ScorerSettings

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

def load_scorer(model, option, alignweights=None):

    # load model parameters and set theano shared variables
    param_list = numpy.load(model).files
    param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
    params = load_params(model, param_list)
    tparams = init_theano_params(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, option)
    inps = [x, x_mask, y, y_mask]
    use_noise.set_value(0.)

    if alignweights:
        logging.debug("Save weight mode ON, alignment matrix will be saved.")
        outputs = [cost, opt_ret['dec_alphas']]
        f_log_probs = theano.function(inps, outputs)
    else:
        f_log_probs = theano.function(inps, cost)

    return f_log_probs

def rescore_model(source_file, target_file, output_file, scorer_settings, options):

    trng = RandomStreams(1234)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        for i, model in enumerate(scorer_settings.models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)
            score, alignment = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalization_alpha=scorer_settings.normalization_alpha, alignweights =alignweights)
            scores.append(score)
            alignments.append(alignment)

        return scores, alignments

    pairs = TextIterator(source_file.name,
                         target_file.name,
                         options[0]['dictionaries'][:-1],
                         options[0]['dictionaries'][-1],
                         n_words_source=options[0]['n_words_src'],
                         n_words_target=options[0]['n_words'],
                         batch_size=scorer_settings.b,
                         maxlen=float('inf'),
                         use_factor=(options[0]['factors'] > 1),
                         sort_by_length=False) #TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, alignments = _score(pairs, scorer_settings.alignweights)

    source_file.seek(0)
    target_file.seek(0)
    source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str,[s[i] for s in scores]))
        if scorer_settings.verbose:
            output_file.write('{0} '.format(line.strip()))
        output_file.write('{0}\n'.format(score_str))

    # optionally save attention weights
    if scorer_settings.alignweights:
        temp_name = output_file.name + ".json"
        with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
            for line in alignments:
                if type(line)==list:
                    for l in line:
                        align_OUT.write(l + "\n")
                else:
                    align_OUT.write(line + "\n")
            # combining the actual source and target words.
            combine_source_target_text_1to1(source_file,
                                            target_file,
                                            output_file.name,
                                            align_OUT)

def main(source_file, target_file, output_file, scorer_settings):
    # load model model_options
    options = []
    for model in scorer_settings.models:
        options.append(load_config(model))
        fill_options(options[-1])
    rescore_model(source_file, target_file, output_file, scorer_settings, options)

if __name__ == "__main__":
    scorer_settings = ScorerSettings(from_console_arguments=True)
    source_file = scorer_settings.source
    target_file = scorer_settings.target
    output_file = scorer_settings.output
    level = logging.DEBUG if scorer_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(source_file, target_file, output_file, scorer_settings)
