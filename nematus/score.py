"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score, and optionally alignment for each pair.
"""

import sys
import argparse
import tempfile

import numpy
import json

from data_iterator import TextIterator
from util import load_dict, load_config
from alignment_util import *
from compat import fill_options

from theano_util import (load_params, init_theano_params)
from nmt import (pred_probs, build_model, prepare_data, init_params)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

def rescore_model(source_file, target_file, saveto, models, options, b, normalize, verbose, alignweights):

    trng = RandomStreams(1234)

    fs_log_probs = []

    for model, option in zip(models, options):

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
            sys.stderr.write("\t*** Save weight mode ON, alignment matrix will be saved.\n")
            outputs = [cost, opt_ret['dec_alphas']]
            f_log_probs = theano.function(inps, outputs)
        else:
            f_log_probs = theano.function(inps, cost)

        fs_log_probs.append(f_log_probs)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        for i, f_log_probs in enumerate(fs_log_probs):
            score, alignment = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalize=normalize, alignweights = alignweights)
            scores.append(score)
            alignments.append(alignment)

        return scores, alignments

    pairs = TextIterator(source_file.name, target_file.name,
                    options[0]['dictionaries'][:-1], options[0]['dictionaries'][-1],
                     n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                     batch_size=b,
                     maxlen=float('inf'),
                     sort_by_length=False) #TODO: sorting by length could be more efficient, but we'd want to resort after

    scores, alignments = _score(pairs, alignweights)

    source_file.seek(0)
    target_file.seek(0)
    source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str,[s[i] for s in scores]))
        if verbose:
            saveto.write('{0} '.format(line.strip()))
        saveto.write('{0}\n'.format(score_str))

    ### optional save weights mode.
    if alignweights:
        ### writing out the alignments.
        temp_name = saveto.name + ".json"
        with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
            for line in all_alignments:
                align_OUT.write(line + "\n")
            ### combining the actual source and target words.
            combine_source_target_text_1to1(source_file, target_file, saveto.name, align_OUT)

def main(models, source_file, nbest_file, saveto, b=80,
         normalize=False, verbose=False, alignweights=False):

    # load model model_options
    options = []
    for model in models:
        options.append(load_config(model))

        fill_options(options[-1])

    rescore_model(source_file, nbest_file, saveto, models, options, b, normalize, verbose, alignweights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', action="store_true",
                        help="Normalize scores by sentence length")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=True,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Source text file")
    parser.add_argument('--target', '-t', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Target text file")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w',required = False,action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <target>.alignment")

    args = parser.parse_args()

    main(args.models, args.source, args.target,
         args.output, b=args.b, normalize=args.n, verbose=args.v, alignweights=args.walign)
