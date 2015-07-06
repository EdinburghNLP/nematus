import numpy
import cPickle as pkl

from nltk.tokenize import wordpunct_tokenize

class TextIterator:
    def __init__(self, source, target, 
                 source_dict, target_dict, 
                 batch_size=128, 
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = open(source, 'r')
        self.target = open(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = 128
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        ii = 0

        try:
            for ii in xrange(self.batch_size):
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = wordpunct_tokenize(ss.decode('utf-8').strip())
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = wordpunct_tokenize(tt.decode('utf-8').strip())
                tt = [self.target_dict[w] if w in self.target_dict else 1 for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target




