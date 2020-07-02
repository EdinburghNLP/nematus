"""Adapted from Nematode: https://github.com/demelin/nematode """

import sys
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import exception
    from . import rnn_inference
    from . import sampler_inputs
    from .transformer import INT_DTYPE, FLOAT_DTYPE
    from . import transformer_inference
except (ModuleNotFoundError, ImportError) as e:
    import exception
    import rnn_inference
    import sampler_inputs
    from transformer import INT_DTYPE, FLOAT_DTYPE
    import transformer_inference


class BeamSearchSampler:
    """Implements beam search with one or more models.

    If there are multiple models, then at each timestep the top k tokens are
    selected according to the sum of the models' log probabilities (where k is
    the beam size).

    The algorithm continues searching until either the maximum translation
    length is reached or until none of the partial translations could go on
    to finish with a better score than the worst finished translation.

    Prior to running the sampler, the placeholders in self.inputs must be
    fed appropriate values (see the SamplerInputs class). Model inputs are fed
    to the model placeholders, as in training.

    The resulting sample can be accessed via the outputs() property, which
    returns a pair of tensors, (sequences, scores). sequences contains target
    vocabulary IDs and has shape (batch_size_x, beam_size, seq_len), where
    seq_len <= max_translation_len is the length of the longest translation.
    scores contains floats representing the length-normalized log
    probabilities. It has the shape (batch_size_x, beam_size).

    TODO Make beam_size a placeholder?

    See also: RandomSampler.
    """

    def __init__(self, models, configs, beam_size):
        """Sets some things up then calls _beam_search() to do the real work.

        Args:
            models: a sequence of RNN or Transformer objects.
            configs: a sequence of model configs (argparse.Namespace objects).
            beam_size: an integer specifying the beam width.
        """
        self._models = models
        self._configs = configs
        self._beam_size = beam_size

        with tf.compat.v1.name_scope('beam_search'):

            # Define placeholders.
            self.inputs = sampler_inputs.SamplerInputs()

            # Create model adapters to get a consistent interface to
            # Transformer and RNN models.
            model_adapters = []
            for i, (model, config) in enumerate(zip(models, configs)):
                with tf.compat.v1.name_scope('model_adapter_{}'.format(i)) as scope:
                    if config.model_type == 'transformer':
                        adapter = transformer_inference.ModelAdapter(
                            model, config, scope)
                    else:
                        assert config.model_type == 'rnn'
                        adapter = rnn_inference.ModelAdapter(
                            model, config, scope)
                    model_adapters.append(adapter)

            # Check that individual models are compatible with each other.
            vocab_sizes = [a.target_vocab_size for a in model_adapters]
            if len(set(vocab_sizes)) > 1:
                raise exception.Error('Cannot ensemble models with different '
                                      'target vocabulary sizes')
            target_vocab_size = vocab_sizes[0]

            # Build the graph to do the actual work.
            sequences, scores = _beam_search(
                model_adapters=model_adapters,
                beam_size=beam_size,
                batch_size_x=self.inputs.batch_size_x,
                max_translation_len=self.inputs.max_translation_len,
                normalization_alpha=self.inputs.normalization_alpha,
                vocab_size=target_vocab_size,
                eos_id=0)

            self._outputs = sequences, scores

    @property
    def outputs(self):
        return self._outputs

    @property
    def models(self):
        return self._models

    @property
    def configs(self):
        return self._configs

    @property
    def beam_size(self):
        return self._beam_size


def _beam_search(model_adapters, beam_size, batch_size_x, max_translation_len,
                 normalization_alpha, vocab_size, eos_id):
    """See description of BeamSearchSampler above.

    Args:
        model_adapters: sequence of ModelAdapter objects.
        beam_size: integer specifying beam width.
        batch_size_x: tf.int32 scalar specifying number of input sentences.
        max_translation_len: tf.int32 scalar specifying max translation length.
        normalization_alpha: tf.float32 scalar specifying alpha parameter for
            length normalization.
        vocab_size: float specifying the target vocabulary size.
        eos_id: integer specifying the vocabulary ID of the EOS symbol.

    Returns:
        A pair of tensors: (sequences, scores). sequences contains vocabulary
        IDs. It has shape (batch_size, len), where len <= max_translation_len
        is the length of the longest translation in the batch. scores contains
        sequnces scores, which are summed probabilities.
    """

    # Encode the input and generate a 1-step decoding function for each model.
    decoding_functions = []
    for adapter in model_adapters:
        encoder_output = adapter.encode()
        func = adapter.generate_decoding_function(encoder_output)
        decoding_functions.append(func)

    # Initialize the timestep counter.
    current_time_step = tf.constant(1)

    # Initialize sequences with <GO>.
    alive_sequences = tf.ones([batch_size_x, beam_size, 1], dtype=INT_DTYPE)
    finished_sequences = tf.ones([batch_size_x, beam_size, 1], dtype=INT_DTYPE)

    # Initialize alive sequence scores.
    alive_scores = tf.zeros([batch_size_x, beam_size])

    # Initialize finished sequence scores to a low value.
    finished_scores = tf.fill([batch_size_x, beam_size], -1. * 1e7)

    # Initialize flags indicating which finished_sequences are really finished.
    finished_eos_flags = tf.fill([batch_size_x, beam_size], False)

    # Initialize memories (i.e. states carried over from the last timestep).
    alive_memories = [ma.generate_initial_memories(batch_size_x, beam_size)
                      for ma in model_adapters]

    # Generate the conditional and body functions for the beam search loop.

    loop_cond = _generate_while_loop_cond_func(max_translation_len)

    loop_body = _generate_while_loop_body_func(model_adapters,
                                               decoding_functions,
                                               max_translation_len,
                                               batch_size_x, beam_size,
                                               vocab_size, eos_id,
                                               normalization_alpha)

    loop_vars = [current_time_step,
                 alive_sequences,
                 alive_scores,
                 finished_sequences,
                 finished_scores,
                 finished_eos_flags,
                 alive_memories]

    shape_invariants=[
        tf.TensorShape([]),                       # timestep
        tf.TensorShape([None, None, None]),       # alive sequences
        alive_scores.get_shape(),                 # alive scores
        tf.TensorShape([None, None, None]),       # finished sequence
        finished_scores.get_shape(),              # finished scores
        finished_eos_flags.get_shape(),           # finished EOS flags
        [adapter.get_memory_invariants(memories)  # alive memories
         for adapter, memories in zip(model_adapters, alive_memories)]]

    # Execute the auto-regressive decoding step via while loop.
    _, alive_sequences, alive_scores, finished_sequences, finished_scores, \
        finished_eos_flags, _ = \
            tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
                cond=loop_cond,
                body=loop_body,
                loop_vars=loop_vars,
                shape_invariants=shape_invariants,
                parallel_iterations=10,
                swap_memory=False))

    alive_sequences.set_shape((None, beam_size, None))
    finished_sequences.set_shape((None, beam_size, None))

    # Account for the case in which no translations terminate in <EOS> for a
    # particular input sentence. In that case, copy the contents of the alive
    # beam for that sentence into the finished beam (sequence + score).
    finished_sequences = tf.compat.v1.where(tf.reduce_any(input_tensor=finished_eos_flags, axis=1),
                                  finished_sequences, alive_sequences)
    # Attention: alive_scores are not length normalized!
    finished_scores = tf.compat.v1.where(tf.reduce_any(input_tensor=finished_eos_flags, axis=1),
                               finished_scores, alive_scores)

    # Truncate finished sequences to remove initial <GO>.
    finished_sequences = finished_sequences[:, :, 1:]

    # Normalize scores. Note that we include the <EOS> token when calculating
    # sequence length.
    seq_len = tf.shape(input=finished_sequences)[2]
    indices = tf.range(seq_len, dtype=tf.int32)
    indices = tf.reshape(indices, [1, 1, seq_len])
    indices = tf.tile(indices, [batch_size_x, beam_size, 1])
    seq_lens = tf.reshape(seq_len, [1, 1, 1])
    seq_lens = tf.tile(seq_lens, [batch_size_x, beam_size, seq_len])
    eos_indices = tf.compat.v1.where(tf.equal(finished_sequences, eos_id),
                           indices, seq_lens)
    lengths = tf.reduce_min(input_tensor=eos_indices+1, axis=2)
    float_lengths = tf.cast(lengths, dtype=tf.float32)
    length_penalties = float_lengths ** normalization_alpha
    finished_scores = finished_scores / length_penalties

    return finished_sequences, finished_scores


def _compute_batch_indices(batch_size_x, beam_size):
    """Generates a matrix of batch indices for the 'merged' beam tensor.

    Each index denotes the batch from which the sequence occupying the same
    relative position as the index within the 'merged' tensor belongs.
    """
    batch_range = tf.range(batch_size_x * beam_size) // beam_size
    batch_index_matrix = tf.reshape(batch_range, [batch_size_x, beam_size])
    return batch_index_matrix


def _gather_top_sequences(model_adapters, all_sequences, all_scores,
                          all_scores_to_gather, all_eos_flags, all_memories,
                          beam_size, batch_size_x, prefix):
    """Selects the top-k sequences from a sequence set."""

    # Obtain indices of the top-k scores within the scores tensor.
    _, top_indices = tf.nn.top_k(all_scores, k=beam_size)

    # Create a lookup-indices tensor for gathering the sequences associated
    # with the top scores.
    batch_index_matrix = _compute_batch_indices(batch_size_x, beam_size)
    gather_coordinates = tf.stack([batch_index_matrix, top_indices], axis=2)

    # Collect top outputs.
    gathered_sequences = tf.gather_nd(all_sequences, gather_coordinates,
                                      name='{:s}_sequences'.format(prefix))

    gathered_scores = tf.gather_nd(all_scores_to_gather, gather_coordinates,
                                   name='{:s}_scores'.format(prefix))

    gathered_eos_flags = tf.gather_nd(all_eos_flags, gather_coordinates,
                                      name='{:s}_eos_flags'.format(prefix))

    gathered_memories = None
    if all_memories is not None:
        gathered_memories = [
            adapter.gather_memories(memories, gather_coordinates)
            for adapter, memories in zip(model_adapters, all_memories)]

    return gathered_sequences, gathered_scores, gathered_eos_flags, \
           gathered_memories


def _generate_while_loop_cond_func(max_translation_len):

    def continue_decoding(curr_time_step, alive_sequences, alive_scores,
                          finished_sequences, finished_scores,
                          finished_eos_flags, alive_memories):
        """Determines whether decoding should continue or terminate."""

        # Check maximum prediction length has not been reached.
        length_criterion = tf.less(curr_time_step, max_translation_len)

        # Otherwise, check if the most likely alive hypothesis is less likely
        # than the least probable completed sequence.

        # Calculate the best possible score of the most probable sequence
        # currently alive.
        highest_alive_score = alive_scores[:, 0]

        # Calculate the score of the least likely sequence currently finished.
        lowest_finished_score = tf.reduce_min(
            input_tensor=finished_scores * tf.cast(finished_eos_flags, FLOAT_DTYPE), axis=1)

        # Account for the case in which none of the sequences in 'finished'
        # have terminated so far; In that case, each of the unfinished
        # sequences is assigned a high negative probability, so that the
        # termination condition is not met.
        tmp = tf.reduce_any(input_tensor=finished_eos_flags, axis=1)
        mask_unfinished = (1. - tf.cast(tmp, dtype=tf.float32)) * (-1. * 1e7)
        lowest_finished_score += mask_unfinished

        # Check is the current highest alive score is lower than the current
        # lowest finished score.
        likelihood_criterion = \
            tf.logical_not(
                tf.reduce_all(
                  input_tensor=tf.greater(lowest_finished_score, highest_alive_score)))

        # Decide whether to continue the decoding process.
        return tf.logical_and(length_criterion, likelihood_criterion)

    return continue_decoding


def _generate_while_loop_body_func(model_adapters, decoding_functions,
                                   max_translation_len, batch_size_x, beam_size,
                                   vocab_size, eos_id, normalization_alpha):

    # Construct an alternate set of 'log probabilities' to use when extending
    # sequences beyond EOS. The value is set very low to ensure that these
    # overgrown sequences are never chosen over incomplete or just-finished
    # sequences.
    tmp = tf.constant(tf.float32.min, shape=[1, 1], dtype=tf.float32)
    eos_log_probs = tf.tile(tmp,
                            multiples=[batch_size_x*beam_size, vocab_size])

    def extend_hypotheses(current_time_step, alive_sequences, alive_scores,
                          alive_memories):
        """Generates top-k extensions of the alive beam candidates."""

        # Get the vocab IDs for this timestep in the order of the model inputs.
        next_ids = alive_sequences[:, :, -1]      # [batch_size_x, beam_size]
        next_ids = tf.transpose(a=next_ids, perm=[1, 0]) # [beam_size, batch_size_x]
        next_ids = tf.reshape(next_ids, [-1])     # [beam_size * batch_size_x]

        # Run the vocab IDs through the decoders and get the log probs for all
        # possible extensions.
        sum_log_probs = None
        for i in range(len(decoding_functions)):

            # Get logits.
            step_logits, alive_memories[i] = decoding_functions[i](
                next_ids, current_time_step, alive_memories[i])

            # Calculate the scores for all possible extensions of alive
            # hypotheses.
            log_probs = tf.nn.log_softmax(step_logits, axis=-1)

            # Add to the log probs from other models.
            if sum_log_probs is None:
                sum_log_probs = log_probs
            else:
                sum_log_probs += log_probs

        # In certain situations, the alive set can legitimately contain
        # sequences that are actually finished. When extending these, we don't
        # care what gets added beyond the EOS, only that the resulting sequence
        # gets a very low score. We give every possible extension the lowest
        # possible probability.
        sum_log_probs = tf.compat.v1.where(tf.equal(next_ids, eos_id),
                                 eos_log_probs,
                                 sum_log_probs)

        # Reshape / transpose to match alive_sequences, alive_scores, etc.
        sum_log_probs = tf.reshape(sum_log_probs,
                                   [beam_size, batch_size_x, vocab_size])
        sum_log_probs = tf.transpose(a=sum_log_probs, perm=[1, 0, 2])

        # Add log probs for this timestep to the full sequence log probs.
        curr_scores = sum_log_probs + tf.expand_dims(alive_scores, axis=2)

        # At first time step, all beams are identical - pick first.
        # At other time steps, flatten all beams.
        flat_curr_scores = tf.cond(pred=tf.equal(current_time_step, 1),
                      true_fn=lambda: curr_scores[:,0],
                      false_fn=lambda: tf.reshape(curr_scores, [batch_size_x, -1]))

        # Select top-k highest scores.
        top_scores, top_ids = tf.nn.top_k(flat_curr_scores, k=beam_size)

        # Determine the beam from which the top-scoring items originate and
        # their identity (i.e. token-ID).
        top_beam_indices = top_ids // vocab_size
        top_ids %= vocab_size

        # Determine the location of top candidates.
        # [batch_size_x, beam_size]
        batch_index_matrix = _compute_batch_indices(batch_size_x, beam_size)
        top_coordinates = tf.stack([batch_index_matrix, top_beam_indices],
                                   axis=2)

        # Extract top decoded sequences.
        # [batch_size_x, beam_size, sent_len]
        top_sequences = tf.gather_nd(alive_sequences, top_coordinates)
        top_sequences = tf.concat([top_sequences,
                                   tf.expand_dims(top_ids, axis=2)],
                                  axis=2)

        # Extract top memories for each model.
        top_memories = [adapter.gather_memories(memories, top_coordinates)
                        for adapter, memories in zip(model_adapters,
                                                     alive_memories)]

        # Check how many of the top sequences have terminated.
        top_eos_flags = tf.equal(top_ids, eos_id)  # [batch_size_x, beam_size]

        return (top_sequences, top_scores, top_eos_flags, top_memories)

    # Define a function to update alive set (part of tf.while_loop body)
    def update_alive(top_sequences, top_scores, top_eos_flags, top_memories):
        """Assembles an updated set of unfinished beam candidates."""

        # Exclude completed sequences from the alive beam by setting their
        # scores to a large negative value.
        selection_scores = top_scores + tf.cast(top_eos_flags, dtype=tf.float32) * (-1.*1e7)

        # Update the alive beam.
        updated_alive_sequences, updated_alive_scores, \
            updated_alive_eos_flags, updated_alive_memories = \
                _gather_top_sequences(model_adapters,
                                      top_sequences,
                                      selection_scores,
                                      top_scores,
                                      top_eos_flags,
                                      top_memories,
                                      beam_size,
                                      batch_size_x,
                                      'alive')

        return updated_alive_sequences, updated_alive_scores, \
               updated_alive_eos_flags, updated_alive_memories

    # Define a function to update finished set (part of tf.while_loop body)
    def update_finished(finished_sequences, finished_scores,
                        finished_eos_flags, top_sequences, top_scores,
                        top_eos_flags):
        """Updates the list of completed translation hypotheses."""

        # Match the length of the 'finished sequences' tensor with the length
        # of the 'finished scores' tensor
        zero_padding = tf.zeros([batch_size_x, beam_size, 1], dtype=INT_DTYPE)
        finished_sequences = tf.concat([finished_sequences, zero_padding],
                                       axis=2)

        # Exclude incomplete sequences from the finished beam by setting their
        # scores to a large negative value
        selection_scores = top_scores \
                           + (1. - tf.cast(top_eos_flags, dtype=tf.float32)) * (-1.*1e7)

        # Combine sequences finished at previous time steps with the top
        # sequences from current time step, as well as their scores and
        # eos-flags, for the selection of a new, most likely, set of finished
        # sequences
        top_finished_sequences = tf.concat([finished_sequences, top_sequences],
                                           axis=1)
        top_finished_scores = tf.concat([finished_scores, selection_scores],
                                        axis=1)
        top_finished_eos_flags = tf.concat([finished_eos_flags, top_eos_flags],
                                           axis=1)
        # Update the finished beam
        updated_finished_sequences, updated_finished_scores, \
            updated_finished_eos_flags, _ = \
                _gather_top_sequences(model_adapters,
                                     top_finished_sequences,
                                     top_finished_scores,
                                     top_finished_scores,
                                     top_finished_eos_flags,
                                     None,
                                     beam_size,
                                     batch_size_x,
                                     'finished')

        return updated_finished_sequences, updated_finished_scores, \
               updated_finished_eos_flags

    def decoding_step(current_time_step, alive_sequences, alive_scores,
                      finished_sequences, finished_scores, finished_eos_flags,
                      alive_memories):
        """Defines a single step of the while loop."""

        # 1. Get the top sequences/ scores/ flags for the current time step
        top_sequences, top_scores, top_eos_flags, top_memories = \
            extend_hypotheses(current_time_step,
                              alive_sequences,
                              alive_scores,
                              alive_memories)

        # 2. Update the alive beam
        alive_sequences, alive_scores, alive_eos_flags, alive_memories = \
            update_alive(top_sequences,
                         top_scores,
                         top_eos_flags,
                         top_memories)

        # 3. Update the finished beam
        finished_sequences, finished_scores, finished_eos_flags = \
            update_finished(finished_sequences,
                            finished_scores,
                            finished_eos_flags,
                            top_sequences,
                            top_scores,
                            top_eos_flags)

        return current_time_step+1, alive_sequences, alive_scores, \
               finished_sequences, finished_scores, finished_eos_flags, \
               alive_memories

    return decoding_step
