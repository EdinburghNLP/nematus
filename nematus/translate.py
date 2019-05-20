#!/usr/bin/env python3

"""Translates a source file using a translation model (or ensemble)."""

import argparse
import logging

import tensorflow as tf

from config import load_config_from_json_file
import inference
import model_loader
import rnn_model
from settings import TranslationSettings
from transformer import Transformer as TransformerModel
from sampling_utils import SamplingUtils


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
        config = load_config_from_json_file(model)
        setattr(config, 'reload', model)
        configs.append(config)

    # Create the model graphs and restore their variables.
    logging.debug("Loading models\n")
    models = []
    for i, config in enumerate(configs):
        with tf.variable_scope("model%d" % i) as scope:
            if config.model_type == "transformer":
                model = TransformerModel(config)
            else:
                model = rnn_model.RNNModel(config)
            saver = model_loader.init_or_restore_variables(config, session,
                                                           ensemble_scope=scope)
            model.sampling_utils = SamplingUtils(settings)
            models.append(model)

    # TODO Ensembling is currently only supported for RNNs, so if
    # TODO len(models) > 1 then check models are all rnn

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
