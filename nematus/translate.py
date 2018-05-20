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
import argparse
import time

from multiprocessing import Process, Queue
from collections import defaultdict
from queue import Empty

from util import load_dict, load_config, seq2words, prepare_data
from compat import fill_options
from hypgraph import HypGraphRenderer
from settings import TranslationSettings

from nmt import create_model, load_dictionaries, read_all_lines

import inference
import exception

class Translation(object):
    """
    Models a translated segment.
    """
    def __init__(self, source_words, target_words, sentence_id=None, score=0, hypothesis_id=None):
        self.source_words = source_words
        self.target_words = target_words
        self.sentence_id = sentence_id
        self.score = score
        self.hypothesis_id = hypothesis_id


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
        self._verbose = settings.verbose
        self._retrieved_translations = defaultdict(dict)

        # load model options
        self._load_model_options()
        # set up queues
        self._init_queues()
        # init worker processes
        self._init_processes()

    def _load_model_options(self):
        """
        Loads config options for each model.
        """

        self._options = []
        for model in self._models:
            config = load_config(model)
            # backward compatibility
            fill_options(config)
            config['reload'] = model
            self._options.append(argparse.Namespace(**config))

        _, _, _, self._num_to_target = load_dictionaries(self._options[0])

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
        for process_id in range(self._num_processes):
            processes[process_id] = Process(
                target=self._start_worker,
                args=(process_id,)
                )
            processes[process_id].start()

        self._processes = processes


    ### MODEL LOADING AND TRANSLATION IN CHILD PROCESS ###



    def _load_models(self, process_id, sess):
        """
        Loads models and returns them
        """
        logging.debug("Process '%s' - Loading models\n" % (process_id))

        import tensorflow as tf
        models = []
        for i, options in enumerate(self._options):
            with tf.name_scope("model%d" % i) as scope:
                model, saver = create_model(options, sess, ensemble_scope=scope)
                models.append(model)

        logging.info("NOTE: Length of translations is capped to {}".format(self._options[0].translation_maxlen))
        return models

    def _start_worker(self, process_id):
        """
        Function executed by each worker once started. Do not execute in
        the parent process.
        """

        # load TF functionality
        import tensorflow as tf
        sess = tf.Session()
        models = self._load_models(process_id, sess)

        # listen to queue in while loop, translate items
        while True:
            input_item = self._input_queue.get()

            if input_item is None:
                break
            idx = input_item.idx
            request_id = input_item.request_id

            output_item = self._translate(process_id, input_item, models, sess)
            self._output_queue.put((request_id, idx, output_item))

        return

    def _translate(self, process_id, input_item, models, sess):
        """
        Actual translation (model sampling).
        """

        # unpack input item attributes
        k = input_item.k
        x = input_item.batch
        #max_ratio = input_item.max_ratio

        y_dummy = numpy.zeros(shape=(len(x),1))
        x, x_mask, _, _ = prepare_data(x, y_dummy, maxlen=None)

        sample = inference.beam_search(models, sess, x, x_mask, k)

        return sample


    ### WRITING TO AND READING FROM QUEUES ###

    def _send_jobs(self, input_, translation_settings):
        """
        """
        source_batches = []

        try:
            batches, idxs = read_all_lines(self._options[0], input_)
        except exception.Error as x:
            logging.error(x.msg)
            for process in self._processes:
                process.terminate()
            sys.exit(1)

        for idx, batch in enumerate(batches):

            input_item = QueueItem(verbose=self._verbose,
                                   k=translation_settings.beam_width,
                                   normalization_alpha=translation_settings.normalization_alpha,
                                   nbest=translation_settings.n_best,
                                   batch=batch,
                                   idx=idx,
                                   request_id=translation_settings.request_id)

            self._input_queue.put(input_item)
            source_batches.append(batch)
        return idx+1, source_batches, idxs

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
                    for midx in range(self._num_processes):
                        if not self._processes[midx].is_alive() and self._processes[midx].exitcode != 0:
                            # kill all other processes and raise exception if one dies
                            self._input_queue.cancel_join_thread()
                            self._output_queue.cancel_join_thread()
                            for idx in range(self._num_processes):
                                self._processes[idx].terminate()
                            logging.error("Translate worker process {0} crashed with exitcode {1}".format(self._processes[midx].pid, self._processes[midx].exitcode))
                            sys.exit(1)
            request_id, idx, output_item = resp
            self._retrieved_translations[request_id][idx] = output_item
            #print self._retrieved_translations

        for idx in range(num_samples):
            yield self._retrieved_translations[request_id][idx]

        # then remove all entries with this request ID from the dictionary
        del self._retrieved_translations[request_id]

    ### EXPOSED TRANSLATION FUNCTIONS ###

    def translate(self, source_segments, translation_settings):
        """
        Returns the translation of @param source_segments.
        """

        logging.info('Translating {0} segments...\n'.format(len(source_segments)))
        start_time = time.time()
        n_batches, source_batches, idxs = self._send_jobs(source_segments, translation_settings)

        n_sent = 0
        outputs = [None]*n_batches
        for i, samples in enumerate(self._retrieve_jobs(n_batches, translation_settings.request_id)):
            outputs[i] = list(samples)
            n_sent += len(samples)
            logging.info('Translated {} sents'.format(n_sent))

        outputs = [beam for batch in outputs for beam in batch]
        outputs = numpy.array(outputs, dtype=numpy.object)
        outputs = outputs[idxs.argsort()]

        translations = []
        for i, beam in enumerate(outputs):
            if translation_settings.normalization_alpha:
                beam = map(lambda sent_cost: (sent_cost[0], sent_cost[1]/len(sent_cost[0]) ** translation_settings.normalization_alpha), beam)
            beam = sorted(beam, key=lambda sent_cost: sent_cost[1])

            if translation_settings.n_best is True:
                n_best_list = []
                for j, (sent, cost) in enumerate(beam):
                    translation = Translation(sentence_id=i,
                                              source_words=source_segments[i],
                                              target_words=seq2words(sent, self._num_to_target, join=False),
                                              score=cost,
                                              hypothesis_id=j)
                    n_best_list.append(translation)
                translations.append(n_best_list)
            else:
                best_hypo, cost = beam[0]
                target_words = seq2words(best_hypo, self._num_to_target)
                translation = Translation(sentence_id=i,
                                            source_words=source_segments[i],
                                            target_words=seq2words(best_hypo, self._num_to_target, join=False),
                                            score=cost)
                translations.append(translation)

        duration = time.time() - start_time
        logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(n_sent, duration, n_sent/duration))

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

        if translation_settings.n_best is True:
            output_file.write(" ||| ".join(output_items) + "\n")
        else:
            output_file.write("\n".join(output_items) + "\n")


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
