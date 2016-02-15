'''
Translates a source file using a translation model.
'''
import sys
import argparse

import numpy
import cPickle as pkl

from multiprocessing import Process, Queue


def translate_model(queue, rqueue, pid, models, options, k, normalize, verbose):

    from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    fs_init = []
    fs_next = []

    for model, option in zip(models, options):

        # allocate model parameters
        params = init_params(option)

        # load model parameters and set theano shared variables
        params = load_params(model, params)
        tparams = init_tparams(params)

        # word index
        f_init, f_next = build_sampler(tparams, option, trng)

        fs_init.append(f_init)
        fs_next.append(f_next)

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(fs_init, fs_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        if verbose:
            sys.stderr.write('{0} - {1}\n'.format(pid,idx))
        seq = _translate(x)

        rqueue.put((idx, seq))

    return


def main(models, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False, verbose=False):

    # load model model_options
    options = []
    for model in args.models:
        with open('%s.pkl' % model, 'rb') as f:
            options.append(pkl.load(f))

    dictionary, dictionary_target = options[0]['dictionaries']

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, models, options, k, normalize, verbose))
        processes[midx].start()

    # utility function
    def _seqs2words(cc):
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict_trg[w])
        return ' '.join(ww)

    def _send_jobs(f):
        for idx, line in enumerate(f):
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()
            x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
            x = map(lambda ii: ii if ii < options[0]['n_words_src'] else 1, x)
            x += [0]
            queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        out_idx = 0
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if verbose and numpy.mod(idx, 10) == 0:
                sys.stderr.write('Sample {0} / {1} Done\n'.format((idx+1), n_samples))
            while out_idx < n_samples and trans[out_idx] != None:
                yield trans[out_idx]
                out_idx += 1

    sys.stderr.write('Translating {0} ...\n'.format(source_file.name))
    n_samples = _send_jobs(source_file)
    _finish_processes()
    for trans in _retrieve_jobs(n_samples):
        saveto.write(_seqs2words(trans) + '\n')

    sys.stderr.write('Done\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5,
                        help="Beam size (default: %(default)s))")
    parser.add_argument('-p', type=int, default=5,
                        help="Number of processes (default: %(default)s))")
    parser.add_argument('-n', action="store_true",
                        help="Normalize scores by sentence length")
    parser.add_argument('-c', action="store_true", help="Character-level")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=True)
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default=sys.stdin, metavar='PATH',
                        help="Input file (default: standard input)")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")

    args = parser.parse_args()

    main(args.models, args.input,
         args.output, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, verbose=args.v)
