'''
Rescoring an n-best list of translations using a translation model.
'''
import sys
import argparse
import tempfile

import numpy
import json
import cPickle as pkl

from data_iterator import TextIterator
from util import load_dict
from alignment_util import *

from nmt import (pred_probs, load_params, build_model, prepare_data,
    init_params, init_tparams)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

def rescore_model(source_file, nbest_file, saveto, models, options, b, normalize, verbose, alignweights):

    trng = RandomStreams(1234)

    fs_log_probs = []

    for model, option in zip(models, options):
        # allocate model parameters
        params = init_params(option)

        # load model parameters and set theano shared variables
        params = load_params(model, params)
        tparams = init_tparams(params)

        trng, use_noise, \
            x, x_mask, y, y_mask, \
            opt_ret, \
            cost = \
            build_model(tparams, option)
        inps = [x, x_mask, y, y_mask]
        use_noise.set_value(0.)

        #### @liucan: added for the command line option.
        if alignweights:
            print "\t*** Save weight mode ON, alignment matrix will be saved."
            outputs = [cost, opt_ret['dec_alphas']]
            f_log_probs = theano.function(inps, outputs)
        else:
            print "\t*** Save weight mode OFF, alignment matrix will not be saved."
            f_log_probs = theano.function(inps, cost)

        fs_log_probs.append(f_log_probs)

    def _score(pairs, alignweights):
        # sample given an input sequence and obtain scores
        scores = []
        all_alignments = []
        for i, f_log_probs in enumerate(fs_log_probs):
            #### @liucan: add this to optional, depending on the choice of file.
            score_this_batch, alignment_this_batch = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalize=normalize, alignweights = alignweights)
            scores.append(score_this_batch)
            all_alignments += alignment_this_batch
        #### @liucan
        return scores, all_alignments


    lines = source_file.readlines()
    nbest_lines = nbest_file.readlines()

    with tempfile.NamedTemporaryFile(prefix='rescore-tmpin') as tmp_in, tempfile.NamedTemporaryFile(prefix='rescore-tmpout') as tmp_out:
    #tempfile.NamedTemporaryFile(prefix="rescore-temp-source-index") as tmp_source_index:
    #### @liucan: added a temporary file to store the index of the source sentences, this is used to print out the alignment matrix.
    #### @liucan: TODO, in a better version, add command line option so people can choose not to store the alignment matrix.
        for line in nbest_lines:
            linesplit = line.split(' ||| ')
            idx = int(linesplit[0])   ##index from the source file. Starting from 0.
            tmp_in.write(lines[idx])
            tmp_out.write(linesplit[1] + '\n')
            #### @liucan: write to file.
            #tmp_source_index.write(str(idx) + "\n")

        tmp_in.seek(0)
        tmp_out.seek(0)
        #tmp_source_index.seek(0)
        pairs = TextIterator(tmp_in.name, tmp_out.name, #tmp_source_index.name,  #### @liucan: need to change too many places, will use the index as a post-processing.
                         options[0]['dictionaries'][0], options[0]['dictionaries'][1],
                         n_words_source=options[0]['n_words_src'], n_words_target=options[0]['n_words'],
                         batch_size=b,
                         maxlen=float('inf'),
                         sort_by_length=False) #TODO: sorting by length could be more efficient, but we'd have to synchronize scores with n-best list after

        if alignweights: #### @liucan
            scores, all_alignments = _score(pairs, alignweights)  #### @liucan: added option.
        else:
            scores = _score(pairs, alignweights)  #### @liucan: added option.
        for i, line in enumerate(nbest_lines):
            score_str = ' '.join(map(str,[s[i] for s in scores]))
            saveto.write('{0} {1}\n'.format(line.strip(), score_str))

        #### @liucan: save the alignments.
        if alignweights:
            ### writing out the alignments.
            with open("alignments.json", "w") as align_OUT:
                for line in all_alignments:
                    align_OUT.write(line + "\n")
    ### combining the actual source and target words.
    combine_source_target_text(source_file, nbest_file)
    #print "In resore.py::"
    #print outputs[1]

def main(models, source_file, nbest_file, saveto, b=80,
         normalize=False, verbose=False, alignweights=False):

    # load model model_options
    options = []
    for model in args.models:
        try:
            with open('%s.json' % model, 'rb') as f:
                options.append(json.load(f))
        except:
            with open('%s.pkl' % model, 'rb') as f:
                options.append(pkl.load(f))
        #hacks for using old models with missing options
        if not 'dropout_embedding' in options[-1]:
            options[-1]['dropout_embedding'] = 0
        if not 'dropout_hidden' in options[-1]:
            options[-1]['dropout_hidden'] = 0
        if not 'dropout_source' in options[-1]:
            options[-1]['dropout_source'] = 0
        if not 'dropout_target' in options[-1]:
            options[-1]['dropout_target'] = 0

    dictionary, dictionary_target = options[0]['dictionaries']

    # load source dictionary and invert
    word_dict = load_dict(dictionary)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    word_dict_trg = load_dict(dictionary_target)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    rescore_model(source_file, nbest_file, saveto, models, options, b, normalize, verbose, alignweights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', action="store_true",
                        help="Normalize scores by sentence length")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=True)
    parser.add_argument('--source', '-s', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Source text file")
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default=sys.stdin, metavar='PATH',
                        help="Input n-best list file (default: standard input)")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w',required = False,action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <input>.alignment")

    args = parser.parse_args()

    main(args.models, args.source, args.input,
         args.output, b=args.b, normalize=args.n, verbose=args.v, alignweights=args.walign)