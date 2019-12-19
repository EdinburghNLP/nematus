import logging
import sys
import time

import numpy
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    import exception
    import util


def translate_batch(session, sampler, x, x_mask, max_translation_len,
                    normalization_alpha):
    """Translate a batch using a RandomSampler or BeamSearchSampler.

    Args:
        session: a TensorFlow session.
        sampler: a BeamSearchSampler or RandomSampler object.
        x: input Tensor with shape (factors, max_seq_len, batch_size).
        x_mask: mask Tensor for x with shape (max_seq_len, batch_size).
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size), sorted by score in best-first
        order.
    """

    x_tiled = numpy.tile(x, reps=[1, 1, sampler.beam_size])
    x_mask_tiled = numpy.tile(x_mask, reps=[1, sampler.beam_size])

    feed_dict = {}

    # Feed inputs to the models.
    for model, config in zip(sampler.models, sampler.configs):
        if config.model_type == 'rnn':
            feed_dict[model.inputs.x] = x_tiled
            feed_dict[model.inputs.x_mask] = x_mask_tiled
        else:
            assert config.model_type == 'transformer'
            # Inputs don't need to be tiled in the Transformer because it
            # checks for different batch sizes in the encoder and decoder and
            # does its own tiling internally at the connection points.
            feed_dict[model.inputs.x] = x
            feed_dict[model.inputs.x_mask] = x_mask
        feed_dict[model.inputs.training] = False

    # Feed inputs to the sampler.
    feed_dict[sampler.inputs.batch_size_x] = x.shape[-1]
    feed_dict[sampler.inputs.max_translation_len] = max_translation_len
    feed_dict[sampler.inputs.normalization_alpha] = normalization_alpha

    # Run the sampler.
    translations, scores = session.run(sampler.outputs, feed_dict=feed_dict)

    assert len(translations) == x.shape[-1]
    assert len(scores) == x.shape[-1]

    # Sort the translations by score. The scores are (optionally normalized)
    # log probs so higher values are better.
    beams = []
    for i in range(len(translations)):
        pairs = zip(translations[i], scores[i])
        beams.append(sorted(pairs, key=lambda pair: pair[1], reverse=True))

    return beams


def translate_file(input_file, output_file, session, sampler, config,
                   max_translation_len, normalization_alpha, nbest=False,
                   minibatch_size=80, maxibatch_size=20):
    """Translates a source file using a RandomSampler or BeamSearchSampler.

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        session: TensorFlow session.
        sampler: BeamSearchSampler or RandomSampler object.
        config: model config.
        max_translation_len: integer specifying maximum translation length.
        normalization_alpha: float specifying alpha parameter for length
            normalization.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
    """

    def translate_maxibatch(maxibatch, num_to_target, num_prev_translated):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(config, maxibatch,
                                                    minibatch_size)
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
            sample = translate_batch(session, sampler, x, x_mask,
                                     max_translation_len, normalization_alpha)
            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]

        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent, num_to_target)
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                line = util.seq2words(best_hypo, num_to_target) + '\n'
                output_file.write(line)

    _, _, _, num_to_target = util.load_dictionaries(config)

    logging.info("NOTE: Length of translations is capped to {}".format(
        max_translation_len))

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
