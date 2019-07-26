#!/usr/bin/env python3

from collections import OrderedDict
import fileinput
import sys

import numpy
import json


def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['<EOS>'] = 0
        worddict['<GO>'] = 1
        worddict['<UNK>'] = 2
        # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+3

        # The JSON RFC requires that JSON text be represented using either
        # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
        # We use UTF-8 regardless of the user's locale settings.
        with open('%s.json'%filename, 'w', encoding='utf-8') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print('Done')

if __name__ == '__main__':
    main()
