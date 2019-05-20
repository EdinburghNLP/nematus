import logging
import sys
import time

import numpy
import tensorflow as tf

import exception
import rnn_inference
import transformer_inference
import util


"""Represents a collection of models that can be used jointly for inference.

RNN and Transformer models are both supported, though they can't be mixed.
For RNNs, search can use multiple models (i.e. an ensemble) but sampling is
limited to a single model. For Transformers, only single models are supported.
Multi-GPU inference is not yet supported.

TODO Multi-GPU inference (i.e. multiple replicas of the same model).
TODO Beam search for Transformer ensembles.
TODO Mixed RNN/Tranformer inference.
TODO Ensemble sampling (is this useful?).
"""
class InferenceModelSet(object):
    def __init__(self, models, configs):
        self._models = models
        self._model_types = [config.model_type for config in configs]
        # Currently only supports RNN ensembles or single Transformer models
        assert len(set(self._model_types)) == 1
        if self._model_types[0] == "transformer":
            assert len(models) == 1
            self._sample_func = transformer_inference.sample
            self._sample_graph_type = transformer_inference.SampleGraph
            self._beam_search_func = transformer_inference.beam_search
            self._beam_search_graph_type = transformer_inference.BeamSearchGraph
        else:
            assert self._model_types[0] == "rnn"
            self._sample_func = rnn_inference.sample
            self._sample_graph_type = rnn_inference.SampleGraph
            self._beam_search_func = rnn_inference.beam_search
            self._beam_search_graph_type = rnn_inference.BeamSearchGraph
        self._cached_sample_graph = None
        self._cached_beam_search_graph = None

    def sample(self, session, x, x_mask):
        # Sampling is not implemented for ensembles, so just use the first
        # model.
        model = self._models[0]
        if self._cached_sample_graph is None:
            self._cached_sample_graph = self._sample_graph_type(model)
        return self._sample_func(session, model, x, x_mask,
                                 self._cached_sample_graph)

    def beam_search(self, session, x, x_mask, beam_size,
                    normalization_alpha=0.0):
        """Beam search using all models contained in this model set.

        If using an ensemble (i.e. more than one model), then at each timestep
        the top k tokens are selected according to the sum of the models' log
        probabilities (where k is the beam size).

        Args:
            session: TensorFlow session.
            x: Numpy array with shape (factors, max_seq_len, batch_size).
            x_mask: Numpy array with shape (max_seq_len, batch_size).
            beam_size: beam width.
            normalization_alpha: length normalization hyperparamter.

        Returns:
            A list of lists of (translation, score) pairs. The outer list
            contains one list for each input sentence in the batch. The inner
            lists contain k elements (where k is the beam size), sorted by
            score in ascending order (i.e. best first, assuming lower scores
            are better).
        """
        def cached_graph_is_usable():
            cached_graph = self._cached_beam_search_graph
            if cached_graph is None:
                return False
            return (cached_graph.beam_size == beam_size
                    and cached_graph.normalization_alpha == normalization_alpha)

        if not cached_graph_is_usable():
            self._cached_beam_search_graph = \
                self._beam_search_graph_type(self._models, beam_size,
                                             normalization_alpha)
        return self._beam_search_func(session, self._models, x, x_mask,
                                      beam_size, normalization_alpha,
                                      self._cached_beam_search_graph)

    def decode(self, session, x, x_mask, beam_size,
               normalization_alpha=0.0):
        """Decode using either beam search or sampling, depending on the translation strategy set in the first model

        Args:
            session: TensorFlow session.
            x: Numpy array with shape (factors, max_seq_len, batch_size).
            x_mask: Numpy array with shape (max_seq_len, batch_size).
            beam_size: beam width (only for beam search).
            normalization_alpha: length normalization hyperparamter (only for beam search).

        Returns:
            A list of lists of (translation, score) pairs. The outer list
            contains one list for each input sentence in the batch. The inner
            lists contain k elements (where k is the beam size), sorted by
            score in ascending order (i.e. best first, assuming lower scores
            are better). When sampling beam size is always one and score is 0.0
        """

        if self._models[0].sampling_utils.translation_strategy == 'beam_search':
            return self.beam_search(session, x, x_mask, beam_size, normalization_alpha)
        elif self._models[0].sampling_utils.translation_strategy == 'sampling':
            samples = self.sample(session, x, x_mask)
            beams = [[(sample, 0.0)] for sample in samples]
            return beams
        else:
            assert False

def translate_file(input_file, output_file, session, models, configs,
                   beam_size=12, nbest=False, minibatch_size=80,
                   maxibatch_size=20, normalization_alpha=1.0):
    """Translates a source file using a translation model (or ensemble).

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        session: TensorFlow session.
        models: list of model objects to use for beam search.
        configs: model configs.
        beam_size: beam width.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
        normalization_alpha: alpha parameter for length normalization.
    """

    def translate_maxibatch(maxibatch, model_set, num_to_target,
                            num_prev_translated):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            model_set: an InferenceModelSet object.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(configs[0], maxibatch,
                                                    minibatch_size)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, configs[0].factors,
                                                maxlen=None)
            sample = model_set.decode(
                session=session,
                x=x,
                x_mask=x_mask,
                beam_size=beam_size,
                normalization_alpha=normalization_alpha)
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

    _, _, _, num_to_target = util.load_dictionaries(configs[0])
    model_set = InferenceModelSet(models, configs)

    logging.info("NOTE: Length of translations is capped to {}".format(
        configs[0].translation_maxlen))

    start_time = time.time()

    num_translated = 0
    maxibatch = []
    while True:
        line = input_file.readline()
        if line == "":
            if len(maxibatch) > 0:
                translate_maxibatch(maxibatch, model_set, num_to_target,
                                    num_translated)
                num_translated += len(maxibatch)
            break
        maxibatch.append(line)
        if len(maxibatch) == (maxibatch_size * minibatch_size):
            translate_maxibatch(maxibatch, model_set, num_to_target,
                                num_translated)
            num_translated += len(maxibatch)
            maxibatch = []

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(
        num_translated, duration, num_translated/duration))
