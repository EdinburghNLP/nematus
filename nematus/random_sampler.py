import sys
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import rnn_inference
    from . import sampler_inputs
    from .transformer import INT_DTYPE, FLOAT_DTYPE
    from . import transformer_inference
except:
    import rnn_inference
    import sampler_inputs
    from transformer import INT_DTYPE, FLOAT_DTYPE
    import transformer_inference


class RandomSampler:
    """Implements random sampling with one or more models.

    Samples translations by randomly drawing one token at a time according to
    the probability distribution over the target vocabulary.

    If beam_size > 1, then multiple translations are sampled for each input
    sentence. Unlike beam search, the translations are sampled independently
    of each other. ('beam_size' is a misnomer in this context, but it
    simplifies things if RandomSampler and BeamSearchSampler have a common
    interface.)

    If there are multiple models, then at each timestep the next token is
    sampled according to the sum of the models' log probabilities.

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

    See also: BeamSearchSampler.
    """

    def __init__(self, models, configs, beam_size):
        """Sets some things up then calls _random_sample() to do the real work.

        Args:
            models: a sequence of RNN or Transformer objects.
            configs: a sequence of model configs (argparse.Namespace objects).
            beam_size: integer specifying the beam width.
        """
        self._models = models
        self._configs = configs
        self._beam_size = beam_size

        with tf.compat.v1.name_scope('random_sampler'):

            # Define placeholders.
            self.inputs = sampler_inputs.SamplerInputs()

            # Create an adapter to get a consistent interface to
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

            # Build the graph to do the actual work.
            sequences, scores = _random_sample(
                model_adapters=model_adapters,
                beam_size=beam_size,
                batch_size_x=self.inputs.batch_size_x,
                max_translation_len=self.inputs.max_translation_len,
                normalization_alpha=self.inputs.normalization_alpha,
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


def _random_sample(model_adapters, beam_size, batch_size_x,
                   max_translation_len, normalization_alpha, eos_id):
    """See description for RandomSampler above.

    Args:
        model_adapters: sequence of ModelAdapter objects.
        beam_size: integer specifying beam width.
        batch_size_x: tf.int32 scalar specifying number of input sentences.
        max_translation_len: tf.int32 scalar specifying max translation length.
        normalization_alpha: tf.float32 scalar specifying alpha parameter for
            length normalization.
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
    sequences = tf.ones([batch_size_x*beam_size, 1], dtype=INT_DTYPE)

    # Initialize sequence scores.
    scores = tf.zeros([batch_size_x*beam_size], dtype=FLOAT_DTYPE)

    # Flags indicating which sequences are finished.
    finished = tf.fill([batch_size_x*beam_size], False)

    # Initialize memories (i.e. states carried over from last timestep.
    memories = [ma.generate_initial_memories(batch_size_x, beam_size)
                for ma in model_adapters]

    # Generate the conditional and body functions for the sampling loop.

    loop_cond = _generate_while_loop_cond_func(max_translation_len)

    loop_body = _generate_while_loop_body_func(model_adapters,
                                               decoding_functions,
                                               batch_size_x, beam_size, eos_id)

    loop_vars = [current_time_step, sequences, scores, memories, finished]

    shape_invariants=[
        tf.TensorShape([]),                             # timestep
        tf.TensorShape([None, None]),                   # sequences
        tf.TensorShape([None]),                         # scores
        [adapter.get_memory_invariants(mems)            # memories
         for adapter, mems in zip(model_adapters, memories)],
        tf.TensorShape([None])]                         # finished

    _, sequences, scores, _, _ = \
        tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond=loop_cond,
                      body=loop_body,
                      loop_vars=loop_vars,
                      shape_invariants=shape_invariants,
                      parallel_iterations=10,
                      swap_memory=False))

    # Truncate sequences to remove leading <GO> tokens.
    sequences = sequences[:, 1:]

    # Normalize scores. Note that we include the <EOS> token when calculating
    # sequence length.
    seq_len = tf.shape(input=sequences)[1]
    indices = tf.range(seq_len, dtype=tf.int32)
    indices = tf.tile(tf.expand_dims(indices, 0), [batch_size_x*beam_size, 1])
    seq_lens = tf.expand_dims(tf.expand_dims(seq_len, 0), 0)
    seq_lens = tf.tile(seq_lens, [batch_size_x*beam_size, seq_len])
    eos_indices = tf.compat.v1.where(tf.equal(sequences, eos_id), indices, seq_lens)
    lengths = tf.reduce_min(input_tensor=eos_indices+1, axis=1)
    float_lengths = tf.cast(lengths, dtype=tf.float32)
    length_penalties = float_lengths ** normalization_alpha
    scores = scores / length_penalties

    # Reshape / transpose to group translations and scores by input sentence.

    sequences = tf.reshape(sequences, [beam_size, batch_size_x, seq_len])
    sequences = tf.transpose(a=sequences, perm=[1,0,2])

    scores = tf.reshape(scores, [beam_size, batch_size_x])
    scores = tf.transpose(a=scores, perm=[1,0])

    return sequences, scores


def _generate_while_loop_cond_func(max_translation_len):

    def continue_decoding(current_time_step, sequences, scores, memories,
                          finished):
        return tf.logical_and(tf.less(current_time_step, max_translation_len),
                              tf.logical_not(tf.reduce_all(input_tensor=finished)))

    return continue_decoding


def _generate_while_loop_body_func(model_adapters, decoding_functions,
                                   batch_size_x, beam_size, eos_id):

    def decoding_step(current_time_step, sequences, scores, memories,
                      finished):

        # Get the target vocabulary IDs for this time step.
        step_ids = sequences[:, -1]

        # Calculate next token probabilities for each model and sum them.
        sum_log_probs = None
        for i in range(len(model_adapters)):
            # Propagate through decoder.
            step_logits, memories[i] = decoding_functions[i](
                step_ids, current_time_step, memories[i])

            # Adjust sampling temperature.
            step_logits = model_adapters[i].model.sampling_utils.adjust_logits(
                step_logits)

            # Calculate log probs for all possible tokens at current time-step.
            model_log_probs = tf.nn.log_softmax(step_logits)

            # Add to summed log probs.
            if sum_log_probs is None:
                sum_log_probs = model_log_probs
            else:
                sum_log_probs += model_log_probs

        # Determine the next token to be added to each sequence.
        next_ids = tf.squeeze(tf.random.categorical(logits=sum_log_probs, num_samples=1,
                                             dtype=INT_DTYPE),
                              axis=1)

        # Collect scores associated with the selected tokens.
        seq_indices = tf.range(batch_size_x * beam_size, dtype=INT_DTYPE)
        score_coordinates = tf.stack([seq_indices, next_ids], axis=1)
        increments =  tf.gather_nd(sum_log_probs, score_coordinates)

        # Add scores to cumulative scores, except for sequences that were
        # already completed before this timestep.
        scores += tf.compat.v1.where(tf.logical_not(finished),
                           increments,
                           tf.zeros([batch_size_x * beam_size]))

        # Extend each sequence with the next token.
        sequences = tf.concat([sequences, tf.expand_dims(next_ids, 1)], 1)

        # Check if sequences have been finished (with a <EOS> token).
        finished |= tf.equal(tf.reduce_prod(input_tensor=sequences - eos_id, axis=1),
                             eos_id)

        return current_time_step+1, sequences, scores, memories, finished

    return decoding_step
