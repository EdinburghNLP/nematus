import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 multi_sentence_separator=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.multi_sentence_separator = multi_sentence_separator

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

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                if self.multi_sentence_separator == None:
                    ss = [ss.split()]
                    tt = [self.target.readline().split()]
                else:
                    ss = [sss.split() for sss in ss.split(self.multi_sentence_separator)]
                    tt = [ttt.split() for ttt in self.target.readline().split(self.multi_sentence_separator)]
                
                if self.skip_empty and (len(ss[0]) == 0 or len(tt[0]) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue
                ss = [ss[0]] + [e for e in ss[1:] if (len(e) > 0) and (len(e) <= self.maxlen)]
                tt = [tt[0]] + [e for e in tt[1:] if (len(e) > 0) and (len(e) <= self.maxlen)]

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t[0]) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    source_multi_sent = self.source_buffer.pop()
                except IndexError:
                    break
                source_multi_sent_word_ids = []
                for source_sent in source_multi_sent:
                    source_sent_word_ids = []
                    for w in source_sent:
                        if self.use_factor:
                            w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                        else:
                            w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                        source_sent_word_ids.append(w)
                    source_multi_sent_word_ids.append(source_sent_word_ids)

                # read from target file and map to word index
                target_multi_sent = self.target_buffer.pop()
                target_multi_sent_word_ids = []
                for target_sent in target_multi_sent:
                    target_sent_word_ids = [self.target_dict[w] if w in self.target_dict else 1 for w in target_sent]
                    if self.n_words_target > 0:
                        target_sent_word_ids = [w_id if w_id < self.n_words_target else 1 for w_id in target_sent_word_ids]
                    target_multi_sent_word_ids.append(target_sent_word_ids)

                if self.multi_sentence_separator == None:
                    source.append(source_multi_sent_word_ids[0])
                    target.append(target_multi_sent_word_ids[0])
                else:
                    source.append(source_multi_sent_word_ids)
                    target.append(target_multi_sent_word_ids)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return source, target
