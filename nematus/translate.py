#!/usr/bin/env python
'''
Translates a source file using a translation model.
'''
import sys
import argparse
import numpy
import json

from util import load_dict, load_config
from compat import fill_options
from hypgraph import HypGraphRenderer
from console import ConsoleInterfaceDefault

from multiprocessing import Process, Queue, Value
from Queue import Empty
from ctypes import c_bool
from cStringIO import StringIO

class Translation(object):
    #TODO move to separate file?
    """
    Models a translated segment.
    """
    def __init(self, source_words, target_words, score=None, alignment=None, target_probs=None, hyp_graph=None):
        self.source_words = source_words
        self.target_words = target_words
        self.score = score
        self.alignment = alignment #TODO: assertion of length?
        self.target_probs = target_probs #TODO: assertion of length?
        self.hyp_graph = hyp_graph

    def get_alignment(self):
        return self.alignment

    def get_alignment_text(self):
        """
        Returns this translation's alignment rendered as a string.
        """
        pass #TODO

    def get_alignment_json(self):
        """
        Returns this translation's alignment in JSON format.
        """
        pass #TODO

    def save_hyp_graph(self, filename, word_idict_trg, detailed=True, highlight_best=True):
        """
        Writes this translation's search graph to disk.
        """
        if self.hyp_graph:
            renderer = HypGraphRenderer(self.hyp_graph)
            renderer.wordify(word_idict_trg)
            renderer.save_png(filename, detailed, highlight_best)
        else:
            pass #TODO: Warning if no search graph has been constructed during decoding?

class QueueItem(object):
    """
    Models items in a queue.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Translator(object):
    """
    Loads trained models and translates.
    """

    def __init__(self, decoder_settings):
        """
        Loads translation models.
        """
        self._models = decoder_settings.models
        self._num_processes = decoder_settings.num_processes
        self._device_list = decoder_settings.device_list
        self._verbose = decoder_settings.verbose
        self._shutdown = Value(c_bool, False)

        # load model options
        self._load_model_options()
        # load and invert dictionaries
        self._build_dictionaries()
        # load translation model
        # work around early binding to GPU device, commented out for now
        # self._load_models()
        # set up FIFO queues and processes
        self._start_queues_processes()

    def _load_model_options(self):
        """
        Loads config options for each model.
        """
        options = []
        for model in self._models:
            options.append(load_config(model))
            # backward compatibility
            fill_options(options[-1])

        self._options = options

    def _build_dictionaries(self):
        """
        Builds and inverts source and target dictionaries, taken
        from the first model since all of them must have the same
        vocabulary.
        """
        dictionaries = self._options[0]['dictionaries']
        dictionaries_source = dictionaries[:-1]
        dictionary_target = dictionaries[-1]

        # load and invert source dictionaries
        word_dicts = []
        word_idicts = []
        for dictionary in dictionaries_source:
            word_dict = load_dict(dictionary)
            if self._options[0]['n_words_src']:
                for key, idx in word_dict.items():
                    if idx >= self._options[0]['n_words_src']:
                        del word_dict[key]
            word_idict = dict()
            for kk, vv in word_dict.iteritems():
                word_idict[vv] = kk
            word_idict[0] = '<eos>'
            word_idict[1] = 'UNK'
            word_dicts.append(word_dict)
            word_idicts.append(word_idict)

        self._word_dicts = word_dicts
        self._word_idicts = word_idicts

        # load and invert target dictionary
        word_dict_trg = load_dict(dictionary_target)
        word_idict_trg = dict()
        for kk, vv in word_dict_trg.iteritems():
            word_idict_trg[vv] = kk
        word_idict_trg[0] = '<eos>'
        word_idict_trg[1] = 'UNK'

        self._word_idict_trg = word_idict_trg

    def _load_models(self):
        """
        Loads models, sets theano shared variables and builds samplers.
        """

        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        from theano import shared

        from nmt import build_sampler
        from theano_util import (load_params, init_theano_params)

        trng = RandomStreams(1234)
        use_noise = shared(numpy.float32(0.))

        fs_init = []
        fs_next = []

        for model, option in zip(self._models, self._options):
            param_list = numpy.load(model).files
            param_list = dict.fromkeys(
                [key for key in param_list if not key.startswith('adam_')], 0)
            params = load_params(model, param_list)
            tparams = init_theano_params(params)

            # always return alignment at this point
            f_init, f_next = build_sampler(
                tparams, option, use_noise, trng, return_alignment=True)

            fs_init.append(f_init)
            fs_next.append(f_next)

        self._trng = trng
        self._fs_init = fs_init
        self._fs_next = fs_next


    def _start_queues_processes(self):
        """
        Sets up input and output queues, starts workers.
        """
        self._input_queue = Queue()
        self._output_queue = Queue()

        self._processes = [None] * self._num_processes

        for process_id in xrange(self._num_processes):
            device_id = ''
            if self._device_list is not None and len(self._device_list) != 0:
                device_id = self._device_list[process_id % len(self._device_list)].strip()


            self._processes[process_id] = Process(target=self._translate,
                                                  args=(process_id, device_id))  # todo: timeout?
            self._processes[process_id].start()


    def _translate(self, process_id, device_id, timeout=1):
        """
        Actual translation (model sampling).
        """

        # if the --device-list argument is set
        if device_id != '':
            import os
            theano_flags = os.environ['THEANO_FLAGS'].split(',')
            exist = False
            for i in xrange(len(theano_flags)):
                if theano_flags[i].strip().startswith('device'):
                    exist = True
                    theano_flags[i] = '%s=%s' % ('device', device_id)
                    break
            if exist == False:
                theano_flags.append('%s=%s' % ('device', device_id))
            os.environ['THEANO_FLAGS'] = ','.join(theano_flags)

        self._load_models()

        while True:
            queue_item = None

            try:
                queue_item = self._input_queue.get(True, timeout=timeout)
            except Empty:
                if self._shutdown.value == True: break

            if queue_item:
                # unpack queue item attributes
                verbose = queue_item.verbose
                normalize = queue_item.normalize
                nbest = queue_item.nbest
                idx = queue_item.idx

                # logging
                if verbose:
                    sys.stderr.write('{0} - {1}\n'.format(process_id, idx))

                # sample given an input sequence and obtain scores
                sample, score, word_probs, alignment, hyp_graph = self._sample(queue_item)

                # normalize scores according to sequence lengths
                if normalize:
                    lengths = numpy.array([len(s) for s in sample])
                    score = score / lengths
                if nbest:
                    item = sample, score, word_probs, alignment, hyp_graph
                else:
                    # return translation with lowest score only
                    sidx = numpy.argmin(score)
                    item = sample[sidx], score[sidx], word_probs[sidx], alignment[sidx], hyp_graph

                # put output item into queue
                self._output_queue.put((idx,item))
        return

    def _sample(self, queue_item):
        """
        Sample from model.
        """
        from nmt import gen_sample

        # unpack queue item attributes
        return_hyp_graph = queue_item.return_hyp_graph
        return_alignment = queue_item.return_alignment
        suppress_unk = queue_item.suppress_unk
        k = queue_item.k
        seq = queue_item.seq

        return gen_sample(self._fs_init, self._fs_next,
                          numpy.array(seq).T.reshape(
                              [len(seq[0]), len(seq), 1]),
                          trng=self._trng, k=k, maxlen=200,
                          stochastic=False, argmax=False,
                          return_alignment=return_alignment,
                          suppress_unk=suppress_unk,
                          return_hyp_graph=return_hyp_graph)

    def _send_jobs(self, input_, translation_settings):
        """
        """
        source_sentences = []
        for idx, line in enumerate(input_):
            if translation_settings.char_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()

            x = []
            for w in words:
                w = [self._word_dicts[i][f] if f in self._word_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                if len(w) != self._options[0]['factors']:
                    sys.stderr.write('Error: expected {0} factors, but input word has {1}\n'.format(self._options[0]['factors'], len(w)))
                    for midx in xrange(self._num_processes):
                        self._processes[midx].terminate()
                    sys.exit(1)
                x.append(w)

            x += [[0]*self._options[0]['factors']]

            queue_item = QueueItem(verbose=self._verbose,
                                   return_hyp_graph=translation_settings.get_search_graph,
                                   return_alignment=translation_settings.get_alignment,
                                   k=translation_settings.beam_width,
                                   suppress_unk=translation_settings.suppress_unk,
                                   normalize=translation_settings.normalize,
                                   nbest=translation_settings.n_best,
                                   seq=x,
                                   idx=idx)

            self._input_queue.put(queue_item)
            source_sentences.append(words)
        return idx+1, source_sentences

    def _retrieve_jobs(self, num_samples, timeout=5):
        """
        """
        trans = [None] * num_samples
        out_idx = 0
        for idx in xrange(num_samples):
            resp = None
            while True:
                try:
                    resp = self._output_queue.get(True, timeout)
                # if queue is empty after 5s, check if processes are still alive
                except Empty:
                    for midx in xrange(self._num_processes):
                        if not self._processes[midx].is_alive() and self._processes[midx].exitcode != 0:
                            # kill all other processes and raise exception if one dies
                            self._input_queue.cancel_join_thread()
                            self._output_queue.cancel_join_thread()
                            for idx in xrange(self._num_processes):
                                self._processes[idx].terminate()
                            sys.stderr.write("Error: translate worker process {0} crashed with exitcode {1}".format(self._processes[midx].pid, self._processes[midx].exitcode))
                            sys.exit(1)
                    # if processes are okay, break
                    if resp is not None: break
            trans[resp[0]] = resp[1]
            if self._verbose and numpy.mod(idx, 10) == 0:
                sys.stderr.write('Sample {0} / {1} Done\n'.format((idx+1), num_samples))
            while out_idx < num_samples and trans[out_idx] != None:
                yield trans[out_idx]
                out_idx += 1

    def _seqs2words(self, cc):
        """
        #todo
        """
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(self._word_idict_trg[w])
        return ' '.join(ww)

    @staticmethod
    def _print_matrix(hyp, file):
        """
        Prints alignment weights for a hypothesis.
        dimension (target_words+1 * source_words+1)
        """
        # each target word has corresponding alignment weights
        for target_word_alignment in hyp:
            # each source hidden state has a corresponding weight
            for w in target_word_alignment:
                print >>file, w,
            print >> file, ""
        print >> file, ""

    @staticmethod
    def _print_matrix_json(hyp, source, target, sid, tid, file):
        source.append("</s>")
        target.append("</s>")
        links = []
        for ti, target_word_alignment in enumerate(hyp):
            for si, w in enumerate(target_word_alignment):
                links.append((target[ti], source[si], str(w), sid, tid))
        json.dump(links, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _print_matrices(mm, file):
        for hyp in mm:
            Translator._print_matrix(hyp, file)
            print >>file, "\n"

    def shutdown(self):
        self._shutdown.value = True

    def translate(self, source_segments, translation_settings):
        """
        Returns the translation of @param source_segments.
        """
        sys.stderr.write('Translating {0} segments...\n'.format(len(source_segments)))
        n_samples, source_sentences = self._send_jobs(source_segments, translation_settings)
        translations = []
        for i, trans in enumerate([trans for trans in self._retrieve_jobs(n_samples)]):
            samples, scores, word_probs, alignment, hyp_graph = trans
            # n-best list
            if translation_settings.n_best > 1:
                order = numpy.argsort(scores)
                n_best_list = []
                for j in order:
                    current_alignment = None if not translation_settings.get_alignment else alignment[j]
                    translation = Translation(source_words=source_segments[i],
                                              target_words=self._seqs2words(samples[j]),
                                              score=scores[j],
                                              alignment=current_alignment,
                                              target_probs=word_probs[j],
                                              hyp_graph=hyp_graph)
                    n_best_list.append(translation)
                translations.append(n_best_list)
            # single-best translation
            else:
                current_alignment = None if not translation_settings.get_alignment else alignment
                translation = Translation(source_words=source_segments[i],
                                          target_words=self._seqs2words(samples),
                                          score=scores,
                                          alignment=current_alignment,
                                          target_probs=word_probs,
                                          hyp_graph=hyp_graph)
                translations.append(translation)
        return translations

    def translate_file(self, input_file, translation_settings):
        """
        """
        with open(input_file) as f:
            source_segments = f.readlines()
        return self.translate(source_segments, translation_settings)


    def translate_string(self, segment, translation_settings):
        """
        Translates a single segment
        """
        if not segment.endswith('\n'):
            segment += '\n'
        source_segments = [segment]
        self.translate(source_segments, translation_settings)

    def translate_list(self, segments, translation_settings):
        """
        Translates a list of segments
        """
        source_segments = [s + '\n' if not s.endswith('\n') else s for s in segments]
        self.translate(source_segments, translation_settings)


def main(input_file, output_file, decoder_settings, translation_settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    translator = Translator(decoder_settings)
    translations = translator.translate_file(input_file, translation_settings)
    #TODO: reproduce writing of probs, alignment etc. from original version
    filecontent = '\n'.join([' '.join(t.target_words) for t in translations])
    if output_file is sys.stdout:
        output_file.write(filecontent)
    else:
        with open(output_file, 'w') as f:
            f.write(filecontent)
    translator.shutdown()


if __name__ == "__main__":
    parser = ConsoleInterfaceDefault()
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    decoder_settings = parser.get_decoder_settings()
    translation_settings = parser.get_translation_settings()
    main(input_file, output_file, decoder_settings, translation_settings)
