#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Translates a source file using a translation model.
'''
import sys
import numpy
import json
import os
import logging

from multiprocessing import Process, Queue
from collections import defaultdict
from Queue import Empty

from util import load_dict, load_config, seqs2words
from compat import fill_options
from hypgraph import HypGraphRenderer
from settings import TranslationSettings

class Translation(object):
    #TODO move to separate file?
    """
    Models a translated segment.
    """
    def __init__(self, source_words, target_words, sentence_id=None, score=0, alignment=None,
                 target_probs=None, hyp_graph=None, hypothesis_id=None):
        self.source_words = source_words
        self.target_words = target_words
        self.sentence_id = sentence_id
        self.score = score
        self.alignment = alignment #TODO: assertion of length?
        self.target_probs = target_probs #TODO: assertion of length?
        self.hyp_graph = hyp_graph
        self.hypothesis_id = hypothesis_id

    def get_alignment(self):
        return self.alignment

    def get_alignment_text(self):
        """
        Returns this translation's alignment rendered as a string.
        Columns in header: sentence id ||| target words ||| score |||
                           source words ||| number of source words |||
                           number of target words
        """
        columns = [
            self.sentence_id,
            " ".join(self.target_words),
            self.score,
            " ".join(self.source_words),
            len(self.source_words) + 1,
            len(self.target_words) + 1
        ]
        header = "{0} ||| {1} ||| {2} ||| {3} ||| {4} {5}\n".format(*columns)

        matrix = []
        for target_word_alignment in self.alignment:
            current_weights = []
            for weight in target_word_alignment:
                current_weights.append(str(weight))
            matrix.append(" ".join(current_weights))

        return header + "\n".join(matrix)

    def get_alignment_json(self, as_string=True):
        """
        Returns this translation's alignment as a JSON serializable object
        (@param as_string False) or a JSON formatted string (@param as_string
        True).
        """

        source_tokens = self.source_words + ["</s>"]
        target_tokens = self.target_words + ["</s>"]

        if self.hypothesis_id is not None:
            tid = self.sentence_id + self.hypothesis_id
        else:
            tid = self.sentence_id
        links = []
        for target_index, target_word_alignment in enumerate(self.alignment):
            for source_index, weight in enumerate(target_word_alignment):
                links.append(
                             (target_tokens[target_index],
                              source_tokens[source_index],
                              str(weight),
                              self.sentence_id,
                              tid)
                             )
        return json.dumps(links, ensure_ascii=False, indent=2) if as_string else links

    def get_target_probs(self):
        """
        Returns this translation's word probabilities as a string.
        """
        return " ".join("{0}".format(prob) for prob in self.target_probs)

    def save_hyp_graph(self, filename, word_idict_trg, detailed=True, highlight_best=True):
        """
        Writes this translation's search graph to disk.
        """
        if self.hyp_graph:
            renderer = HypGraphRenderer(self.hyp_graph)
            renderer.wordify(word_idict_trg)
            renderer.save(filename, detailed, highlight_best)
        else:
            pass #TODO: Warning if no search graph has been constructed during decoding?

class QueueItem(object):
    """
    Models items in a queue.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Translator(object):

    def __init__(self, settings):
        """
        Loads translation models.
        """
        self._models = settings.models
        self._num_processes = settings.num_processes
        self._device_list = settings.device_list
        self._verbose = settings.verbose
        self._retrieved_translations = defaultdict(dict)

        # load model options
        self._load_model_options()
        # load and invert dictionaries
        self._build_dictionaries()
        # set up queues
        self._init_queues()
        # init worker processes
        self._init_processes()

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

    def _init_queues(self):
        """
        Sets up shared queues for inter-process communication.
        """
        self._input_queue = Queue()
        self._output_queue = Queue()

    def shutdown(self):
        """
        Executed from parent process to terminate workers,
        method: "poison pill".
        """
        for process in self._processes:
            self._input_queue.put(None)

    def _init_processes(self):
        """
        Starts child (worker) processes.
        """
        processes = [None] * self._num_processes
        for process_id in xrange(self._num_processes):
            deviceid = ''
            if self._device_list is not None and len(self._device_list) != 0:
                deviceid = self._device_list[process_id % len(self._device_list)].strip()
            processes[process_id] = Process(
                target=self._start_worker,
                args=(process_id, deviceid)
                )
            processes[process_id].start()

        self._processes = processes


    ### MODEL LOADING AND TRANSLATION IN CHILD PROCESS ###

    def _load_theano(self):
        """
        Loads models, sets theano shared variables and builds samplers.
        This entails irrevocable binding to a specific GPU.
        """

        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        from theano import shared

        from nmt import (build_sampler, gen_sample)
        from theano_util import (numpy_floatX, load_params, init_theano_params)

        trng = RandomStreams(1234)
        use_noise = shared(numpy_floatX(0.))

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

        return trng, fs_init, fs_next, gen_sample

    def _set_device(self, device_id):
        """
        Modifies environment variable to change the THEANO device.
        """
        if device_id != '':
            try:
                theano_flags = os.environ['THEANO_FLAGS'].split(',')
                exist = False
                for i in xrange(len(theano_flags)):
                    if theano_flags[i].strip().startswith('device'):
                        exist = True
                        theano_flags[i] = '%s=%s' % ('device', device_id)
                        break
                if exist is False:
                    theano_flags.append('%s=%s' % ('device', device_id))
                os.environ['THEANO_FLAGS'] = ','.join(theano_flags)
            except KeyError:
                # environment variable does not exist at all
                os.environ['THEANO_FLAGS'] = 'device=%s' % device_id

    def _load_models(self, process_id, device_id):
        """
        Modifies environment variable to change the THEANO device, then loads
        models and returns them.
        """
        logging.debug("Process '%s' - Loading models on device %s\n" % (process_id, device_id))

        # modify environment flag 'device'
        self._set_device(device_id)

        # build and return models
        return self._load_theano()

    def _start_worker(self, process_id, device_id):
        """
        Function executed by each worker once started. Do not execute in
        the parent process.
        """
        # load theano functionality
        trng, fs_init, fs_next, gen_sample = self._load_models(process_id, device_id)

        # listen to queue in while loop, translate items
        while True:
            input_item = self._input_queue.get()

            if input_item is None:
                break
            idx = input_item.idx
            request_id = input_item.request_id

            output_item = self._translate(process_id, input_item, trng, fs_init, fs_next, gen_sample)
            self._output_queue.put((request_id, idx, output_item))

        return

    def _translate(self, process_id, input_item, trng, fs_init, fs_next, gen_sample):
        """
        Actual translation (model sampling).
        """

        # unpack input item attributes
        normalization_alpha = input_item.normalization_alpha
        nbest = input_item.nbest
        idx = input_item.idx

        # logging
        logging.debug('{0} - {1}\n'.format(process_id, idx))

        # sample given an input sequence and obtain scores
        sample, score, word_probs, alignment, hyp_graph = self._sample(input_item, trng, fs_init, fs_next, gen_sample)

        # normalize scores according to sequence lengths
        if normalization_alpha:
            adjusted_lengths = numpy.array([len(s) ** normalization_alpha for s in sample])
            score = score / adjusted_lengths
        if nbest is True:
            output_item = sample, score, word_probs, alignment, hyp_graph
        else:
            # return translation with lowest score only
            sidx = numpy.argmin(score)
            output_item = sample[sidx], score[sidx], word_probs[
                sidx], alignment[sidx], hyp_graph

        return output_item

    def _sample(self, input_item, trng, fs_init, fs_next, gen_sample):
        """
        Sample from model.
        """

        # unpack input item attributes
        return_hyp_graph = input_item.return_hyp_graph
        return_alignment = input_item.return_alignment
        suppress_unk = input_item.suppress_unk
        k = input_item.k
        seq = input_item.seq
        max_ratio = input_item.max_ratio

        maxlen = 200 #TODO: should be configurable
        if max_ratio:
          maxlen = int(max_ratio * len(seq))

        return gen_sample(fs_init, fs_next,
                          numpy.array(seq).T.reshape(
                              [len(seq[0]), len(seq), 1]),
                          trng=trng, k=k, maxlen=maxlen,
                          stochastic=False, argmax=False,
                          return_alignment=return_alignment,
                          suppress_unk=suppress_unk,
                          return_hyp_graph=return_hyp_graph)


    ### WRITING TO AND READING FROM QUEUES ###

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
                    logging.warning('Expected {0} factors, but input word has {1}\n'.format(self._options[0]['factors'], len(w)))
                    for midx in xrange(self._num_processes):
                        self._processes[midx].terminate()
                    sys.exit(1)
                x.append(w)

            x += [[0]*self._options[0]['factors']]

            input_item = QueueItem(verbose=self._verbose,
                                   return_hyp_graph=translation_settings.get_search_graph,
                                   return_alignment=translation_settings.get_alignment,
                                   k=translation_settings.beam_width,
                                   suppress_unk=translation_settings.suppress_unk,
                                   normalization_alpha=translation_settings.normalization_alpha,
                                   nbest=translation_settings.n_best,
                                   max_ratio=translation_settings.max_ratio,
                                   seq=x,
                                   idx=idx,
                                   request_id=translation_settings.request_id)

            self._input_queue.put(input_item)
            source_sentences.append(words)
        return idx+1, source_sentences

    def _retrieve_jobs(self, num_samples, request_id, timeout=5):
        """
        """
        while len(self._retrieved_translations[request_id]) < num_samples:
            resp = None
            while resp is None:
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
                            logging.error("Translate worker process {0} crashed with exitcode {1}".format(self._processes[midx].pid, self._processes[midx].exitcode))
                            sys.exit(1)
            request_id, idx, output_item = resp
            self._retrieved_translations[request_id][idx] = output_item
            #print self._retrieved_translations

        for idx in xrange(num_samples):
            yield self._retrieved_translations[request_id][idx]

        # then remove all entries with this request ID from the dictionary
        del self._retrieved_translations[request_id]

    ### EXPOSED TRANSLATION FUNCTIONS ###

    def translate(self, source_segments, translation_settings):
        """
        Returns the translation of @param source_segments.
        """
        logging.info('Translating {0} segments...\n'.format(len(source_segments)))
        n_samples, source_sentences = self._send_jobs(source_segments, translation_settings)

        translations = []
        for i, trans in enumerate(self._retrieve_jobs(n_samples, translation_settings.request_id)):

            samples, scores, word_probs, alignment, hyp_graph = trans
            # n-best list
            if translation_settings.n_best is True:
                order = numpy.argsort(scores)
                n_best_list = []
                for j in order:
                    current_alignment = None if not translation_settings.get_alignment else alignment[j]
                    translation = Translation(sentence_id=i,
                                              source_words=source_sentences[i],
                                              target_words=seqs2words(samples[j], self._word_idict_trg, join=False),
                                              score=scores[j],
                                              alignment=current_alignment,
                                              target_probs=word_probs[j],
                                              hyp_graph=hyp_graph,
                                              hypothesis_id=j)
                    n_best_list.append(translation)
                translations.append(n_best_list)
            # single-best translation
            else:
                current_alignment = None if not translation_settings.get_alignment else alignment
                translation = Translation(sentence_id=i,
                                          source_words=source_sentences[i],
                                          target_words=seqs2words(samples, self._word_idict_trg, join=False),
                                          score=scores,
                                          alignment=current_alignment,
                                          target_probs=word_probs,
                                          hyp_graph=hyp_graph)
                translations.append(translation)
        return translations

    def translate_file(self, input_object, translation_settings):
        """
        """
        source_segments = input_object.readlines()
        return self.translate(source_segments, translation_settings)


    def translate_string(self, segment, translation_settings):
        """
        Translates a single segment
        """
        if not segment.endswith('\n'):
            segment += '\n'
        source_segments = [segment]
        return self.translate(source_segments, translation_settings)

    def translate_list(self, segments, translation_settings):
        """
        Translates a list of segments
        """
        source_segments = [s + '\n' if not s.endswith('\n') else s for s in segments]
        return self.translate(source_segments, translation_settings)

    ### FUNCTIONS FOR WRITING THE RESULTS ###

    def write_alignment(self, translation, translation_settings):
        """
        Writes alignments to a file.
        """
        output_file = translation_settings.output_alignment
        if translation_settings.json_alignment:
            output_file.write(translation.get_alignment_json() + "\n")
        else:
            output_file.write(translation.get_alignment_text() + "\n\n")

    def write_translation(self, output_file, translation, translation_settings):
        """
        Writes a single translation to a file or STDOUT.
        """
        output_items = []
        # sentence ID only for nbest
        if translation_settings.n_best is True:
            output_items.append(str(translation.sentence_id))

        # translations themselves
        output_items.append(" ".join(translation.target_words))

        # write scores for nbest?
        if translation_settings.n_best is True:
            output_items.append(str(translation.score))

        # write probabilities?
        if translation_settings.get_word_probs:
            output_items.append(translation.get_target_probs())

        if translation_settings.n_best is True:
            output_file.write(" ||| ".join(output_items) + "\n")
        else:
            output_file.write("\n".join(output_items) + "\n")

        # write alignments to file?
        if translation_settings.get_alignment:
            self.write_alignment(translation, translation_settings)

        # construct hypgraph?
        if translation_settings.get_search_graph:
            translation.save_hyp_graph(
                                       translation_settings.search_graph_filename,
                                       self._word_idict_trg,
                                       detailed=True,
                                       highlight_best=True
            )


    def write_translations(self, output_file, translations, translation_settings):
        """
        Writes translations to a file or STDOUT.
        """
        if translation_settings.n_best is True:
            for nbest_list in translations:
                for translation in nbest_list:
                    self.write_translation(output_file, translation, translation_settings)
        else:
            for translation in translations:
                self.write_translation(output_file, translation, translation_settings)

def main(input_file, output_file, translation_settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    translator = Translator(translation_settings)
    translations = translator.translate_file(input_file, translation_settings)
    translator.write_translations(output_file, translations, translation_settings)

    logging.info('Done')
    translator.shutdown()


if __name__ == "__main__":
    # parse console arguments
    translation_settings = TranslationSettings(from_console_arguments=True)
    input_file = translation_settings.input
    output_file = translation_settings.output
    # start logging
    level = logging.DEBUG if translation_settings.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(input_file, output_file, translation_settings)
