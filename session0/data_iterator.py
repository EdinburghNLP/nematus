import numpy
import cPickle as pkl
import gzip

from nltk.tokenize import wordpunct_tokenize

class TextIterator:
    def __init__(self, source,
                 source_dict,
                 batch_size=128, 
                 maxlen=100,
                 n_words_source=-1):
        if source.endswith('.gz'):
            self.source = gzip.open(source, 'r')
        else:
            self.source = open(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        ii = 0

        try:
            while True:
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = wordpunct_tokenize(ss.decode('utf-8').strip())
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                if len(ss) > self.maxlen:
                    continue

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source




