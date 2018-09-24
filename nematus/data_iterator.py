import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class FileWrapper(object):
    def __init__(self, fname):
        self.pos = 0
        self.lines = fopen(fname).readlines()
        self.lines = numpy.array(self.lines, dtype=numpy.object)
    def __iter__(self):
        return self
    def next(self):
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
        return self.next()
    def shuffle_lines(self, perm):
        self.lines = self.lines[perm]
        self.pos = 0
    def __len__(self):
        return len(self.lines)

class MultiTargetIterator:
    """Bitext iterator for use with Johnson et al (2017)-style multilingual NMT
       (i.e. where first token of source is <2xx> indicating that target
       language is xx). Produces batches in which the source language can
       vary, but the target language is a single language."""
    def __init__(self, source, target,
                 source_langs, target_langs,
                 source_dicts, target_dicts,
                 embedding_map=None,
                 batch_size=128,
                 maxlen=100,
                 source_vocab_sizes=-1,
                 target_vocab_size=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 token_batch_size=0,
                 keep_data_in_memory=False):

        assert embedding_map != None or \
            (len(source_langs) == 1 and len(target_langs) == 1)

        corpora, embedding_maps, self.corpus_lengths = \
            self._split_corpus_by_target_lang(source, target, embedding_map,
                                              target_langs)
        assert len(corpora) == len(target_langs)
        assert len(embedding_maps) == len(target_langs)
        assert len(self.corpus_lengths) == len(target_langs)
        self.subiterators = []
        for i, (source, target) in enumerate(corpora):
            it = TextIterator(source=source,
                              target=target,
                              source_langs=source_langs,
                              source_dicts=source_dicts,
                              target_dict=target_dicts[i],
                              embedding_map=embedding_maps[i],
                              batch_size=batch_size,
                              maxlen=maxlen,
                              source_vocab_sizes=source_vocab_sizes,
                              target_vocab_size=target_vocab_size,
                              skip_empty=skip_empty,
                              shuffle_each_epoch=shuffle_each_epoch,
                              sort_by_length=sort_by_length,
                              use_factor=use_factor,
                              maxibatch_size=maxibatch_size,
                              token_batch_size=token_batch_size,
                              keep_data_in_memory=keep_data_in_memory)
            self.subiterators.append(it)
        self.dead = set()
        self.selection_probs = self._compute_selection_probs(
            self.corpus_lengths, self.dead)

    def __iter__(self):
        return self

    def next(self):
        while len(self.dead) < len(self.subiterators):
            i = numpy.random.choice(len(self.subiterators),
                                    p=self.selection_probs)
            try:
                source, target = self.subiterators[i].next()
                return source, target, i
            except StopIteration:
                self.dead.add(i)
                self.selection_probs = self._compute_selection_probs(
                    self.corpus_lengths, self.dead)
        self.reset()
        raise StopIteration

    def reset(self):
        for subiter in self.subiterators:
            subiter.reset()
        self.dead = set()
        self.selection_probs = self._compute_selection_probs(
            self.corpus_lengths, self.dead)

    def _split_corpus_by_target_lang(self, source, target, embedding_map,
                                     target_langs):
        assert embedding_map != None or len(target_langs) == 1
        def determine_target_lang(source_line):
            first_token = source_line.split()[0]
            return first_token[2:4]
        source_names, target_names = [], []
        source_files, target_files = [], []
        embedding_map_names = []
        embedding_map_files = []
        target_lang_reverse_map = {}
        for i, lang in enumerate(target_langs):
            source_names.append(source + "." + lang)
            target_names.append(target + "." + lang)
            source_files.append(open(source_names[-1], "w"))
            target_files.append(open(target_names[-1], "w"))
            target_lang_reverse_map[lang] = i
            if embedding_map == None:
                embedding_map_names.append(None)
            else:
                embedding_map_names.append(embedding_map + "." + lang)
                embedding_map_files.append(open(embedding_map_names[-1], "w"))
        corpus_lengths = [0] * len(target_langs)
        source_file = open(source)
        target_file = open(target)
        embedding_map_file = None if embedding_map == None \
                                  else open(embedding_map)
        while True:
            source_line = source_file.readline()
            if source_line == "":
                break
            target_line = target_file.readline()
            if embedding_map_file == None:
                lang = target_langs[0]
            else:
                lang_pair_line = embedding_map_file.readline()
                _, lang = lang_pair_line.split()
            i = target_lang_reverse_map[lang]
            source_files[i].write(source_line)
            target_files[i].write(target_line)
            if embedding_map_file != None:
                embedding_map_files[i].write(lang_pair_line)
            corpus_lengths[i] += 1
        for i, lang in enumerate(target_langs):
            source_files[i].close()
            target_files[i].close()
            if embedding_map_file != None:
                embedding_map_files[i].close()
        corpora = zip(source_names, target_names)
        return corpora, embedding_map_names, corpus_lengths

    def _compute_selection_probs(self, corpus_lengths, dead):
        probs = [0.0] * len(corpus_lengths)
        total_len = sum([corpus_lengths[i]
                        for i in range(len(corpus_lengths)) if i not in dead])
        if total_len == 0:
            return probs
        for i in range(len(corpus_lengths)):
            if i not in dead:
                probs[i] = float(corpus_lengths[i]) / float(total_len)
        return probs


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_langs,
                 source_dicts, target_dict,
                 embedding_map=None,
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
                 keep_data_in_memory=False):
        assert embedding_map != None or len(source_langs) == 1
        self.source_lang_reverse_map = {}
        for i, lang in enumerate(source_langs):
            self.source_lang_reverse_map[lang] = i
        if keep_data_in_memory:
            self.source, self.target = FileWrapper(source), FileWrapper(target)
            if embedding_map == None:
                self.embedding_map = None
            else:
                self.embedding_map = FileWrapper(embedding_map)
            if shuffle_each_epoch:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
                if self.embedding_map != None:
                    self.embedding_map.shuffle_lines(r)
        elif shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            if embedding_map == None:
                self.source, self.target = shuffle.main(
                    [self.source_orig, self.target_orig], temporary=True)
                self.embedding_map = None
            else:
                self.embedding_map_orig = embedding_map
                self.source, self.target, self.embedding_map = shuffle.main(
                    [self.source_orig, self.target_orig,
                    self.embedding_map_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            if embedding_map == None:
                self.embedding_map = None
            else:
                self.embedding_map = fopen(embedding_map, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

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
                    for key, idx in d.items():
                        if idx >= vocab_size:
                            del d[key]

        if self.target_vocab_size != None and self.target_vocab_size > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.target_vocab_size:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.source_lang_index_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            if self.keep_data_in_memory:
                r = numpy.random.permutation(len(self.source))
                self.source.shuffle_lines(r)
                self.target.shuffle_lines(r)
                if self.embedding_map != None:
                    self.embedding_map.shuffle_lines(r)
            else:
                if self.embedding_map != None:
                    self.source, self.target, self.embedding_map = shuffle.main(
                        [self.source_orig, self.target_orig,
                        self.embedding_map_orig], temporary=True)
                else:
                    self.source, self.target = shuffle.main(
                        [self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            if self.embedding_map != None:
                self.embedding_map.seek(0)

    def next(self):
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
        assert len(self.source_buffer) == len(self.source_lang_index_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                if self.embedding_map == None:
                    src_lang_idx = 0
                else:
                    src_lang, tgt_lang = self.embedding_map.readline().split()
                    src_lang_idx = self.source_lang_reverse_map[src_lang]

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                self.source_lang_index_buffer.append(src_lang_idx)
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
                _xbuf = [self.source_lang_index_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                self.source_lang_index_buffer = _xbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                self.source_lang_index_buffer.reverse()

        def adjust_source_indices(w, src_lang_idx, vocab_sizes):
            num_factors = len(w)
            num_langs = len(vocab_sizes) / num_factors
            offsets = [0] * num_factors
            for i in range(num_factors):
                for j in range(src_lang_idx):
                    offsets[i] += vocab_sizes[i * num_langs + j]
            return [index + offsets[i] for i, index in enumerate(w)]

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                src_lang_idx = self.source_lang_index_buffer.pop()

                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    w = adjust_source_indices(w, src_lang_idx,
                                              self.source_vocab_sizes)
                    tmp.append(w)
                ss_indices = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt_indices = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.target_vocab_size != None:
                    tt_indices = [w if w < self.target_vocab_size else 1 for w in tt_indices]

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
                        self.source_lang_index_buffer.append(src_lang_idx)

                        break

                else:
                    if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True

        return source, target
