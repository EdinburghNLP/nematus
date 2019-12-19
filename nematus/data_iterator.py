import sys
import numpy
import logging

import gzip

import subprocess

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .util import load_dict
    from . import shuffle
except (ModuleNotFoundError, ImportError) as e:
    from util import load_dict
    import shuffle

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding="UTF-8")
    return open(filename, mode, encoding="UTF-8")

class FileWrapper(object):
    def __init__(self, fname):
        self.pos = 0
        self.lines = fopen(fname).readlines()
        self.lines = numpy.array(self.lines, dtype=numpy.object)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos >= len(self.lines):
            raise StopIteration
        l = self.lines[self.pos]
        self.pos += 1
        return l
    def reset(self):
        self.pos = 0
    def seek(self, pos):
        assert pos == 0
        self.pos = 0
    def readline(self):
        return next(self)
    def shuffle_lines(self, perm):
        self.lines = self.lines[perm]
        self.pos = 0
    def __len__(self):
        return len(self.lines)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 model_type,
                 batch_size=128,
                 maxlen=100,
                 source_vocab_sizes=None,
                 target_vocab_size=None,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 token_batch_size=0,
                 keep_data_in_memory=False,
                 preprocess_script=None):
        self.preprocess_script = preprocess_script
        self.source_orig = source
        self.target_orig = target
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
        if keep_data_in_memory:
            self.source, self.target = FileWrapper(source), FileWrapper(target)
            if shuffle_each_epoch:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
        elif shuffle_each_epoch:
            self.source, self.target = shuffle.jointly_shuffle_files(
                [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict, model_type))
        self.target_dict = load_dict(target_dict, model_type)

        # Determine the UNK value for each dictionary (the value depends on
        # which version of build_dictionary.py was used).

        def determine_unk_val(d):
            if '<UNK>' in d and d['<UNK>'] == 2:
                return 2
            return 1

        self.source_unk_vals = [determine_unk_val(d)
                                for d in self.source_dicts]
        self.target_unk_val = determine_unk_val(self.target_dict)


        self.keep_data_in_memory = keep_data_in_memory
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.source_vocab_sizes = source_vocab_sizes
        self.target_vocab_size = target_vocab_size

        self.token_batch_size = token_batch_size

        if self.source_vocab_sizes != None:
            assert len(self.source_vocab_sizes) == len(self.source_dicts)
            for d, vocab_size in zip(self.source_dicts, self.source_vocab_sizes):
                if vocab_size != None and vocab_size > 0:
                    for key, idx in list(d.items()):
                        if idx >= vocab_size:
                            del d[key]

        if self.target_vocab_size != None and self.target_vocab_size > 0:
            for key, idx in list(self.target_dict.items()):
                if idx >= self.target_vocab_size:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.preprocess_script:
            logging.info("Executing external preprocessing script...")
            proc = subprocess.Popen(self.preprocess_script)
            proc.wait()
            logging.info("done")
            if self.keep_data_in_memory:
                self.source, self.target = FileWrapper(self.source_orig), FileWrapper(self.target_orig)
            else:
                self.source = fopen(self.source_orig, 'r')
                self.target = fopen(self.target_orig, 'r')
        if self.shuffle:
            if self.keep_data_in_memory:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
            else:
                self.source, self.target = shuffle.jointly_shuffle_files(
                    [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        longest_source = 0
        longest_target = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by source/target buffer length
            if self.sort_by_length:
                tlen = numpy.array([max(len(s),len(t)) for (s,t) in zip(self.source_buffer,self.target_buffer)])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        def lookup_token(t, d, unk_val):
            return d[t] if t in d else unk_val

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
                        w = [lookup_token(f, self.source_dicts[i],
                                          self.source_unk_vals[i])
                             for (i, f) in enumerate(w.split('|'))]
                    else:
                        w = [lookup_token(w, self.source_dicts[0],
                                          self.source_unk_vals[0])]
                    tmp.append(w)
                ss_indices = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt_indices = [lookup_token(w, self.target_dict,
                                           self.target_unk_val) for w in tt]
                if self.target_vocab_size != None:
                    tt_indices = [w if w < self.target_vocab_size
                                    else self.target_unk_val
                                  for w in tt_indices]

                source.append(ss_indices)
                target.append(tt_indices)
                longest_source = max(longest_source, len(ss_indices))
                longest_target = max(longest_target, len(tt_indices))

                if self.token_batch_size:
                    if len(source)*longest_source > self.token_batch_size or \
                        len(target)*longest_target > self.token_batch_size:
                        # remove last sentence pair (that made batch over-long)
                        source.pop()
                        target.pop()
                        self.source_buffer.append(ss)
                        self.target_buffer.append(tt)

                        break

                else:
                    if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True

        return source, target
