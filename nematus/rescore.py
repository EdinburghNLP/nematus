#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Rescoring an n-best list of translations using a translation model.
'''
import sys
import argparse
import tempfile
import logging

import numpy

from data_iterator import TextIterator
from util import load_config
from alignment_util import combine_source_target_text
from compat import fill_options

from theano_util import (floatX, numpy_floatX, load_params, init_theano_params)
from nmt import (pred_probs, build_model, prepare_data)
from score import load_scorer
from settings import RescorerSettings

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

def rescore_model(source_file, nbest_file, output_file, rescorer_settings, options):

    trng = RandomStreams(1234)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        for i, model in enumerate(rescorer_settings.models):
            f_log_probs = load_scorer(model, options[i], alignweights=alignweights)
            score, alignment = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalization_alpha=rescorer_settings.normalization_alpha, alignweights = alignweights)
            scores.append(score)
            alignments.append(alignment)

        return scores, alignments

    lines = source_file.readlines()
    nbest_lines = nbest_file.readlines()

    if rescorer_settings.alignweights: ### opening the temporary file.
        temp_name = output_file.name + ".json"
        align_OUT = tempfile.NamedTemporaryFile(prefix=temp_name)

    with tempfile.NamedTemporaryFile(prefix='rescore-tmpin') as tmp_in, tempfile.NamedTemporaryFile(prefix='rescore-tmpout') as tmp_out:
        for line in nbest_lines:
            linesplit = line.split(' ||| ')
            idx = int(linesplit[0])   ##index from the source file. Starting from 0.
            tmp_in.write(lines[idx])
            tmp_out.write(linesplit[1] + '\n')

        tmp_in.seek(0)
        tmp_out.seek(0)
        pairs = TextIterator(tmp_in.name,
                             tmp_out.name,
                             options[0]['dictionaries'][:-1],
                             options[0]['dictionaries'][1],
                             n_words_source=options[0]['n_words_src'],
                             n_words_target=options[0]['n_words'],
                             batch_size=rescorer_settings.b,
                             maxlen=float('inf'),
                             use_factor=(options[0]['factors'] > 1),
                             sort_by_length=False) #TODO: sorting by length could be more efficient, but we'd have to synchronize scores with n-best list after


        scores, alignments = _score(pairs, rescorer_settings.alignweights)

        for i, line in enumerate(nbest_lines):
            score_str = ' '.join(map(str,[s[i] for s in scores]))
            output_file.write('{0} {1}\n'.format(line.strip(), score_str))

        ### optional save weights mode.
        if rescorer_settings.alignweights:
            for line in alignments:
                if type(line)==list:
                    for l in line:
                        align_OUT.write(l + "\n")
                else:
                    align_OUT.write(line + "\n")
    if rescorer_settings.alignweights:
        combine_source_target_text(source_file, nbest_file, output_file.name, align_OUT)
        align_OUT.close()

def main(source_file, nbest_file, output_file, rescorer_settings):
    # load model model_options
    options = []
    for model in rescorer_settings.models:
        options.append(load_config(model))
        fill_options(options[-1])
    rescore_model(source_file, nbest_file, output_file, rescorer_settings, options)

if __name__ == "__main__":
    rescorer_settings = RescorerSettings(from_console_arguments=True)
    source_file = rescorer_settings.source
    nbest_file = rescorer_settings.input
    output_file = rescorer_settings.output
    level = logging.DEBUG if rescorer_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(source_file, nbest_file, output_file, rescorer_settings)
