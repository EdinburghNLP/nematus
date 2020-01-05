#!/usr/bin/env python3

"""Translates a source file using a translation model (or ensemble)."""

import sys
import logging
if __name__ == '__main__':
    # Parse console arguments.
    from settings import TranslationSettings
    settings = TranslationSettings(from_console_arguments=True)
    # Set the logging level. This needs to be done before the tensorflow
    # module is imported.
    level = logging.DEBUG if settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

import argparse

import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .beam_search_sampler import BeamSearchSampler
    from .config import load_config_from_json_file
    from .exponential_smoothing import ExponentialSmoothing
    from . import model_loader
    from .random_sampler import RandomSampler
    from . import rnn_model
    from .sampling_utils import SamplingUtils
    from .transformer import Transformer as TransformerModel
    from . import translate_utils
except (ModuleNotFoundError, ImportError) as e:
    from beam_search_sampler import BeamSearchSampler
    from config import load_config_from_json_file
    from exponential_smoothing import ExponentialSmoothing
    import model_loader
    from random_sampler import RandomSampler
    import rnn_model
    from sampling_utils import SamplingUtils
    from transformer import Transformer as TransformerModel
    import translate_utils


def main(settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    # Create the TensorFlow session.
    g = tf.Graph()
    with g.as_default():
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.allow_soft_placement = True
        session = tf.compat.v1.Session(config=tf_config)

        # Load config file for each model.
        configs = []
        for model in settings.models:
            config = load_config_from_json_file(model)
            setattr(config, 'reload', model)
            configs.append(config)

        # Create the model graphs.
        logging.debug("Loading models\n")
        models = []
        for i, config in enumerate(configs):
            with tf.compat.v1.variable_scope("model%d" % i) as scope:
                if config.model_type == "transformer":
                    model = TransformerModel(config)
                else:
                    model = rnn_model.RNNModel(config)
                model.sampling_utils = SamplingUtils(settings)
                models.append(model)

        # Add smoothing variables (if the models were trained with smoothing).
        #FIXME Assumes either all models were trained with smoothing or none were.
        if configs[0].exponential_smoothing > 0.0:
            smoothing = ExponentialSmoothing(configs[0].exponential_smoothing)

        # Restore the model variables.
        for i, config in enumerate(configs):
            with tf.compat.v1.variable_scope("model%d" % i) as scope:
                _ = model_loader.init_or_restore_variables(config, session,
                                                       ensemble_scope=scope)

        # Swap-in the smoothed versions of the variables.
        if configs[0].exponential_smoothing > 0.0:
            session.run(fetches=smoothing.swap_ops)

        # FIXME Should be an option in settings
        max_translation_len = configs[0].translation_maxlen

        # Create a BeamSearchSampler / RandomSampler.
        if settings.translation_strategy == 'beam_search':
            sampler = BeamSearchSampler(models, configs, settings.beam_size)
        else:
            assert settings.translation_strategy == 'sampling'
            sampler = RandomSampler(models, configs, settings.beam_size)

        # Warn about the change from neg log probs to log probs for the RNN.
        if settings.n_best:
            model_types = [config.model_type for config in configs]
            if 'rnn' in model_types:
                logging.warn('n-best scores for RNN models have changed from '
                             'positive to negative (as of commit 95793196...). '
                             'If you are using the scores for reranking etc, then '
                             'you may need to update your scripts.')

        # Translate the source file.
        translate_utils.translate_file(
            input_file=settings.input,
            output_file=settings.output,
            session=session,
            sampler=sampler,
            config=configs[0],
            max_translation_len=max_translation_len,
            normalization_alpha=settings.normalization_alpha,
            nbest=settings.n_best,
            minibatch_size=settings.minibatch_size,
            maxibatch_size=settings.maxibatch_size)


if __name__ == "__main__":
    main(settings)
