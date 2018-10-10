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

from settings import RescorerSettings
from score import score_model

import tensorflow as tf


def rescore(source_file, nbest_file, output_file, rescorer_settings, options):

    lines = source_file.readlines()
    nbest_lines = nbest_file.readlines()

    # create plain text file for scoring
    with tempfile.NamedTemporaryFile(prefix='rescore-tmpin') as tmp_in, tempfile.NamedTemporaryFile(prefix='rescore-tmpout') as tmp_out:
        for line in nbest_lines:
            linesplit = line.split(' ||| ')
            idx = int(linesplit[0])   ##index from the source file. Starting from 0.
            tmp_in.write(lines[idx])
            tmp_out.write(linesplit[1] + '\n')

        tmp_in.seek(0)
        tmp_out.seek(0)

        scores = score_model(tmp_in, tmp_out, rescorer_settings, options)

    for i, line in enumerate(nbest_lines):
        score_str = ' '.join(map(str,[s[i] for s in scores]))
        output_file.write('{0} {1}\n'.format(line.strip(), score_str))

def main(source_file, nbest_file, output_file, rescorer_settings):
    # load model model_options
    options = []
    for model in rescorer_settings.models:
        options.append(load_config(model))
        fill_options(options[-1])
        options[-1]['reload'] = model
        options[-1] = argparse.Namespace(**options[-1])

    rescore(source_file, nbest_file, output_file, rescorer_settings, options)

if __name__ == "__main__":
    rescorer_settings = RescorerSettings(from_console_arguments=True)
    source_file = rescorer_settings.source
    nbest_file = rescorer_settings.input
    output_file = rescorer_settings.output
    level = logging.DEBUG if rescorer_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(source_file, nbest_file, output_file, rescorer_settings)
