#!/usr/bin/env python

"""Translates a source file using a translation model (or ensemble)."""

import argparse
import logging
import sys
import time

import numpy
import tensorflow as tf

import compat
import exception
import inference
from model import StandardModel
import nmt
from settings import TranslationSettings
import util


def translate_file(input_file, output_file, source_lang, target_lang, session,
                   models, config, beam_size=12, nbest=False, minibatch_size=80,
                   maxibatch_size=20, normalization_alpha=1.0):
    """Translates a source file using a translation model (or ensemble).

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        source_lang: ...
        target_lang: ...
        session: TensorFlow session.
        models: list of model objects to use for ensemble beam search.
        config: model config (must be valid for all models).
        beam_size: beam width.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
        normalization_alpha: alpha parameter for length normalization.
    """

    def normalize(sent, cost):
        return (sent, cost / (len(sent) ** normalization_alpha))

    def translate_maxibatch(maxibatch, num_to_target, num_prev_translated):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = nmt.read_all_lines(config, maxibatch,
                                                   minibatch_size, source_lang)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                                maxlen=None)
            sample = inference.beam_search(models, session, x, x_mask,
                                           beam_size, target_lang)
            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]

        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if normalization_alpha:
                beam = map(lambda (sent, cost): normalize(sent, cost), beam)
            beam = sorted(beam, key=lambda (sent, cost): cost)
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent,
                                                 num_to_target[target_lang])
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                line = util.seq2words(best_hypo,
                                      num_to_target[target_lang]) + '\n'
                output_file.write(line)

    _, _, _, num_to_target = nmt.load_dictionaries(config)

    logging.info("NOTE: Length of translations is capped to {}".format(
        config.translation_maxlen))

    start_time = time.time()

    num_translated = 0
    maxibatch = []
    while True:
        line = input_file.readline()
        if line == "":
            if len(maxibatch) > 0:
                translate_maxibatch(maxibatch, num_to_target, num_translated)
                num_translated += len(maxibatch)
            break
        maxibatch.append(line)
        if len(maxibatch) == (maxibatch_size * minibatch_size):
            translate_maxibatch(maxibatch, num_to_target, num_translated)
            num_translated += len(maxibatch)
            maxibatch = []

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(
        num_translated, duration, num_translated/duration))


def main():
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    # Parse console arguments.
    settings = TranslationSettings(from_console_arguments=True)

    # Start logging.
    level = logging.DEBUG if settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Create the TensorFlow session.
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    session = tf.Session(config=tf_config)

    # Load config file for each model.
    configs = []
    for model in settings.models:
        config = util.load_config(model)
        compat.fill_options(config)
        config['reload'] = model
        configs.append(argparse.Namespace(**config))

    # Determine source and target language IDs.
    source_lang, target_lang = None, None
    for i, lang in enumerate(configs[0].source_embedding_ids):
        if lang == settings.source_embedding_id:
            source_lang = i
            break
    if source_lang == None:
        assert False
    for i, lang in enumerate(configs[0].target_embedding_ids):
        if lang == settings.target_embedding_id:
            target_lang = i
            break
    if target_lang == None:
        assert False

    # Create the model graphs and restore their variables.
    logging.debug("Loading models\n")
    models = []
    for i, config in enumerate(configs):
        with tf.variable_scope("model%d" % i) as scope:
            model = StandardModel(config)
            saver = nmt.init_or_restore_variables(config, session,
                                                  ensemble_scope=scope)
            models.append(model)

    # Translate the source file.
    translate_file(input_file=settings.input,
                   output_file=settings.output,
                   source_lang=source_lang,
                   target_lang=target_lang,
                   session=session,
                   models=models,
                   config=configs[0],
                   beam_size=settings.beam_width,
                   nbest=settings.n_best,
                   minibatch_size=settings.b,
                   maxibatch_size=settings.maxibatch_size,
                   normalization_alpha=settings.normalization_alpha)


if __name__ == "__main__":
    main()
