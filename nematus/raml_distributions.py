'''
Functions to generate exponentiated payoff distributions for RAML
'''

import numpy
import scipy.misc as misc

def hamming_distance_distribution(sentence_length, vocab_size, tau=1.0):
    #based on https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
    c = numpy.zeros(sentence_length)
    for edit_dist in xrange(sentence_length):
        n_edits = misc.comb(sentence_length, edit_dist)
        #reweight
        c[edit_dist] = numpy.log(n_edits) + edit_dist * numpy.log(vocab_size)
        c[edit_dist] = c[edit_dist] - edit_dist / tau - edit_dist / tau * numpy.log(vocab_size)

    c = numpy.exp(c)
    c /= numpy.sum(c)
    return c


def edit_distance_distribution(sentence_length, vocab_size, tau=1.0):
    #from https://gist.github.com/norouzi/8c4d244922fa052fa8ec18d8af52d366
    c = numpy.zeros(sentence_length)
    for edit_dist in xrange(sentence_length):
        n_edits = 0
        for n_substitutes in xrange(min(sentence_length, edit_dist)+1):
            n_insert = edit_dist - n_substitutes
            current_edits = misc.comb(sentence_length, n_substitutes, exact=False) * \
                misc.comb(sentence_length+n_insert-n_substitutes, n_insert, exact=False)
            n_edits += current_edits
        c[edit_dist] = numpy.log(n_edits) + edit_dist * numpy.log(vocab_size)
        c[edit_dist] = c[edit_dist] - edit_dist / tau - edit_dist / tau * numpy.log(vocab_size)

    c = numpy.exp(c)
    c /= numpy.sum(c)
    return c
