"""Utility functions."""

import pickle as pkl
import json
import logging
import numpy
import sys

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
except (ModuleNotFoundError, ImportError) as e:
    import exception

# batch preparation
def prepare_data(seqs_x, seqs_y, n_factors, maxlen=None):
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
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = list(zip(*s_x))
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


def load_dict(filename, model_type):
    try:
        # build_dictionary.py writes JSON files as UTF-8 so assume that here.
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        # FIXME Should we be assuming UTF-8?
        with open(filename, 'r', encoding='utf-8') as f:
            d = pkl.load(f)

    # The transformer model requires vocab dictionaries to use the new style
    # special symbols. If the dictionary looks like an old one then tell the
    # user to update it.
    if model_type == 'transformer' and ("<GO>" not in d or d["<GO>"] != 1):
        logging.error('you must update \'{}\' for use with the '
                      '\'transformer\' model type. Please re-run '
                      'build_dictionary.py to generate a new vocabulary '
                      'dictionary.'.format(filename))
        sys.exit(1)

    return d


def seq2words(seq, inverse_dictionary, join=True):
    seq = numpy.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_dictionary],
                             join)

def factoredseq2words(seq, inverse_dictionaries, join=True):
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    eos_reached = False
    for i, w in enumerate(seq):
        if eos_reached:
            break
        factors = []
        for j, f in enumerate(w):
            if f == 0:
                eos_reached = True
                break
                # This assert has been commented out because it's possible for
                # non-zero values to follow zero values for Transformer models.
                # TODO Check why this happens
                #assert (i == len(seq) - 1) or (seq[i+1][j] == 0), \
                #       ('Zero not at the end of sequence', seq)
            elif f in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][f])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words

def reverse_dict(dictt):
    keys, values = list(zip(*list(dictt.items())))
    r_dictt = dict(list(zip(values, keys)))
    return r_dictt


def load_dictionaries(config):
    model_type = config.model_type
    source_to_num = [load_dict(d, model_type) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict, model_type)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target


def read_all_lines(config, sentences, batch_size):
    source_to_num, _, _, _ = load_dictionaries(config)

    if config.source_vocab_sizes != None:
        assert len(config.source_vocab_sizes) == len(source_to_num)
        for d, vocab_size in zip(source_to_num, config.source_vocab_sizes):
            if vocab_size != None and vocab_size > 0:
                for key, idx in list(d.items()):
                    if idx >= vocab_size:
                        del d[key]

    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            if config.factors == 1:
                w = [source_to_num[0][w] if w in source_to_num[0] else 2]
            else:
                w = [source_to_num[i][f] if f in source_to_num[i] else 2
                                         for (i,f) in enumerate(w.split('|'))]
                if len(w) != config.factors:
                    raise exception.Error(
                        'Expected {0} factors, but input word has {1}\n'.format(
                            config.factors, len(w)))
            line.append(w)
        lines.append(line)
    lines = numpy.array(lines)
    lengths = numpy.array([len(l) for l in lines])
    idxs = lengths.argsort()
    lines = lines[idxs]

    #merge into batches
    batches = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batches.append(batch)

    return batches, idxs
