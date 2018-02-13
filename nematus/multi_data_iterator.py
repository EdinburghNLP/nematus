import numpy

import gzip

import shuffle
from util import load_dict

import math

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class MultilingualTextIterator:
    """Multilingual bitext iterator."""
    def __init__(self, sources, targets,
                 source_dicts, target_dict,
                 language_source_target,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        
        self.source_dicts = []
        
        self.finished_files = []
        self.sources = []
        self.targets = []
        self.sources_orig = []
        self.targets_orig = []
        self.target_languages = []
        self.triple_id = 0
        
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)
        
        #Here we go!
        for lst in language_source_target:
            self.target_languages.append(lst[0]) 
            if shuffle_each_epoch:
                self.sources_orig.append(lst[1]) 
                self.targets_orig.append(lst[2]) 
                a,b = shuffle.main([self.sources_orig[-1], self.targets_orig[-1]], temporary=True)
                self.sources.append(a) 
                self.targets.append(b) 
                self.finished_files.append(False) 
            else:
                self.sources.append(fopen(lst[1], 'r')) 
                self.targets.append(fopen(lst[2], 'r')) 
                self.finished_files.append(False) 
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

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

    def stop(self):
    
        stop = True
        for f in self.finished_files:
            if f == False:
                stop = False
        if stop == True:
            for x in range(0, len(self.finished_files)):
                self.finished_files[x] = False
            raise StopIteration
    
    def reset(self, x):
        
        if self.shuffle:
            self.sources[x], self.targets[x] = shuffle.main([self.sources_orig[x], self.targets_orig[x]], temporary=True)
        else:
            self.sources[x].seek(0)
            self.targets[x].seek(0)
        self.finished_files[x] = True
        self.stop()

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            # self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            
            for x in range(0, len(self.sources)):
                for k_ in xrange(int(math.floor(self.k/len(self.sources)))):
                    ss = "<2" + self.target_languages[x] + "> " + self.sources[x].readline()
                    if ss == "<2" + self.target_languages[x] + "> ":
                        self.reset(x)
                        break
                    tt = self.targets[x].readline()
                    if tt == "":
                        self.reset(x)
                        break

                    self.source_buffer.append(ss.strip().split())
                    self.target_buffer.append(tt.strip().split())

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
            # self.reset()
            raise StopIteration

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
                    w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
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
