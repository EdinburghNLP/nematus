'''
Utility functions
'''

import sys
import json
import cPickle as pkl

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seqs2words(seq, inverse_target_dictionary, join=True):
    words = []
    for w in seq:
        if w == 0:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append('UNK')
    return ' '.join(words) if join else words
