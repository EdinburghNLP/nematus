import numpy

import gzip

import shuffle
from util import load_dict

import math

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DomainInterpolatorTextIterator:
    """Bitext iterator with domain interpolation."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 indomain_source='', indomain_target='',
                 interpolation_rate=0.1,
                 use_factor=False,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
            self.indomain_source_orig = indomain_source
            self.indomain_target_orig = indomain_target
            self.indomain_source, self.indomain_target = shuffle.main([self.indomain_source_orig, self.indomain_target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            self.indomain_source = fopen(indomain_source, 'r')
            self.indomain_target = fopen(indomain_target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

        self.interpolation_rate = interpolation_rate
        self.cur_interpolation_rate = self.interpolation_rate
        self.indomain_k = int(math.ceil(self.cur_interpolation_rate * self.k))
        self.outdomain_k = self.k - self.indomain_k

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def indomain_reset(self):
        if self.shuffle:
            self.indomain_source, self.indomain_target = shuffle.main([self.indomain_source_orig, self.indomain_target_orig], temporary=True)
        else:
            self.indomain_source.seek(0)
            self.indomain_target.seek(0)

    def adjust_domain_interpolation_rate(self, interpolation_rate):
        # discard sentences in buffers
        self.source_buffer = []
        self.target_buffer = []
        # adjust rate
        self.cur_interpolation_rate = interpolation_rate
        self.indomain_k = int(math.ceil(self.cur_interpolation_rate * self.k))
        self.outdomain_k = self.k - self.indomain_k
        
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            #raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.outdomain_k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
            for k_ in xrange(self.indomain_k):
                indomain_error = False
                try:
                    ss = self.indomain_source.readline()
                    tt = self.indomain_target.readline()
                except IOError:
                    indomain_error = True
                if (ss == "") or (tt == "") or indomain_error:
                    self.indomain_reset()
                    raise StopIteration
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            #raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
