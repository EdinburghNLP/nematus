'''
Encode a source file using the encoder of a trained translation model.
'''
import argparse

import numpy
import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def encode_model(queue, rqueue, pid, model, options):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng)

    def _encode(seq):
        # encode the source sentence
        code = f_init(numpy.array(seq).reshape([len(seq), 1]))[1]
        return code

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx
        cod = _encode(x)

        rqueue.put((idx, cod))

    return


def main(model, dictionary, source_file, saveto, 
         n_process=5, chr_level=False):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=encode_model,
            args=(queue, rqueue, midx, model, options,))
        processes[midx].start()

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        codes = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            codes[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return codes

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file)
    codes = numpy.array(_retrieve_jobs(n_samples))
    _finish_processes()
    if not saveto.endswith('npy'):
        saveto = saveto + '.npy'
    numpy.save(saveto, codes)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('-p', type=int, default=4)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.source,
         args.saveto, n_process=args.p, chr_level=args.c)
