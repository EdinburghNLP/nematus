#!/usr/bin/env python3
"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score.
"""

import logging
if __name__ == '__main__':
    # Parse console arguments.
    from settings import ScorerSettings
    scorer_settings = ScorerSettings(from_console_arguments=True)
    # Set the logging level. This needs to be done before the tensorflow
    # module is imported.
    level = logging.DEBUG if scorer_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

import argparse
import sys
import tempfile

import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .config import load_config_from_json_file
    from .data_iterator import TextIterator
    from .exponential_smoothing import ExponentialSmoothing
    from . import model_loader
    from . import rnn_model
    from . import train
    from . import transformer
except (ModuleNotFoundError, ImportError) as e:
    from config import load_config_from_json_file
    from data_iterator import TextIterator
    from exponential_smoothing import ExponentialSmoothing
    import model_loader
    import rnn_model
    import train
    import transformer



# FIXME pass in paths not file objects, since we need to know the paths anyway
def calc_scores(source_file, target_file, scorer_settings, configs):
    """Calculates sentence pair scores using each of the specified models.

    By default (when scorer_settings.normalization_alpha is 0.0), the score
    is the sentence-level cross entropy, otherwise it's a normalized version.

    Args:
        source_file: file object for file containing source sentences.
        target_file: file object for file containing target sentences.
        scorer_settings: a ScorerSettings object.
        configs: a list of Namespace objects specifying the model configs.

    Returns:
        A list of lists of floats. The outer list contains one list for each
        model (in the same order given by configs). The inner list contains
        one score for each sentence pair.
    """
    scores = []
    for config in configs:
        g = tf.Graph()
        with g.as_default():
            tf_config = tf.compat.v1.ConfigProto()
            tf_config.allow_soft_placement = True
            with tf.compat.v1.Session(config=tf_config) as sess:

                logging.info('Building model...')

                # Create the model graph.
                if config.model_type == 'transformer':
                    model = transformer.Transformer(config)
                else:
                    model = rnn_model.RNNModel(config)

                # Add smoothing variables (if the model was trained with
                # smoothing).
                if config.exponential_smoothing > 0.0:
                    smoothing = ExponentialSmoothing(
                        config.exponential_smoothing)

                # Restore the model variables.
                saver = model_loader.init_or_restore_variables(config, sess)

                # Swap-in the smoothed versions of the variables (if present).
                if config.exponential_smoothing > 0.0:
                    sess.run(fetches=smoothing.swap_ops)

                text_iterator = TextIterator(
                    source=source_file.name,
                    target=target_file.name,
                    source_dicts=config.source_dicts,
                    target_dict=config.target_dict,
                    model_type=config.model_type,
                    batch_size=scorer_settings.minibatch_size,
                    maxlen=float('inf'),
                    source_vocab_sizes=config.source_vocab_sizes,
                    target_vocab_size=config.target_vocab_size,
                    use_factor=(config.factors > 1),
                    sort_by_length=False)

                ce_vals, _ = train.calc_cross_entropy_per_sentence(
                    sess,
                    model,
                    config,
                    text_iterator,
                    normalization_alpha=scorer_settings.normalization_alpha)

                scores.append(ce_vals)
    return scores


def write_scores(source_file, target_file, scores, output_file, scorer_settings):

    source_file.seek(0)
    target_file.seek(0)
    source_lines = source_file.readlines()
    target_lines = target_file.readlines()

    for i, line in enumerate(target_lines):
        score_str = ' '.join(map(str,[s[i] for s in scores]))
        if scorer_settings.verbose:
            output_file.write('{0} '.format(line.strip()))
        output_file.write('{0}\n'.format(score_str))


def main(source_file, target_file, output_file, scorer_settings):
    # load model model_options
    configs = []
    for model in scorer_settings.models:
        config = load_config_from_json_file(model)
        setattr(config, 'reload', model)
        configs.append(config)

    scores = calc_scores(source_file, target_file, scorer_settings, configs)
    write_scores(source_file, target_file, scores, output_file, scorer_settings)


if __name__ == "__main__":
    main(source_file=scorer_settings.source,
         target_file=scorer_settings.target,
         output_file=scorer_settings.output,
         scorer_settings=scorer_settings)
