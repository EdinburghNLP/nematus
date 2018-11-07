#!/usr/bin/env python

"""Translates a source file using a translation model (or ensemble)."""

import argparse
import logging

import tensorflow as tf

import compat
import inference
import model_loader
import rnn_model
from settings import TranslationSettings
import util


def main(settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
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

    # Create the model graphs and restore their variables.
    logging.debug("Loading models\n")
    models = []
    for i, config in enumerate(configs):
        with tf.variable_scope("model%d" % i) as scope:
            model = rnn_model.RNNModel(config)
            saver = model_loader.init_or_restore_variables(config, session,
                                                           ensemble_scope=scope)
            models.append(model)

    # Translate the source file.
    inference.translate_file(input_file=settings.input,
                             output_file=settings.output,
                             session=session,
                             models=models,
                             configs=configs,
                             beam_size=settings.beam_size,
                             nbest=settings.n_best,
                             minibatch_size=settings.minibatch_size,
                             maxibatch_size=settings.maxibatch_size,
                             normalization_alpha=settings.normalization_alpha)


if __name__ == "__main__":
    # Parse console arguments.
    settings = TranslationSettings(from_console_arguments=True)
    main(settings)
