import numpy
import tensorflow as tf


def sample(session, model, x, x_mask, graph=None):
    """Randomly samples translations from a RNNModel.

    Args:
        session: TensorFlow session.
        model: a RNNModel object.
        x: Numpy array with shape (factors, max_seq_len, batch_size).
        x_mask: Numpy array with shape (max_seq_len, batch_size).
        graph: a SampleGraph object (to allow reuse if sampling repeatedly).

    Returns:
        A list of NumPy arrays (one for each input sentence in x).
    """
    feed_dict = {model.inputs.x: x, model.inputs.x_mask: x_mask}
    if graph is None:
        graph = SampleGraph(model)
    sampled_ys = session.run(graph.outputs, feed_dict=feed_dict)
    sampled_ys = sampled_ys.T
    samples = []
    for sample in sampled_ys:
        sample = numpy.trim_zeros(list(sample), trim='b')
        sample.append(0)
        samples.append(sample)
    assert len(samples) == x.shape[-1]
    return samples


def beam_search(session, models, x, x_mask, beam_size,
                normalization_alpha=0.0, graph=None):
    """Beam search using one or more RNNModels..

    If using an ensemble (i.e. more than one model), then at each timestep
    the top k tokens are selected according to the sum of the models' log
    probabilities (where k is the beam size).

    Args:
        session: TensorFlow session.
        models: a list of RNNModel objects.
        x: Numpy array with shape (factors, max_seq_len, batch_size).
        x_mask: Numpy array with shape (max_seq_len, batch_size).
        beam_size: beam width.
        normalization_alpha: length normalization hyperparamter.
        graph: a BeamSearchGraph (to allow reuse if searching repeatedly).

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size), sorted by score in ascending
        order (i.e. best first, assuming lower scores are better).
    """
    def normalize(sent, cost):
        return (sent, cost / (len(sent) ** normalization_alpha))

    x_repeat = numpy.repeat(x, repeats=beam_size, axis=-1)
    x_mask_repeat = numpy.repeat(x_mask, repeats=beam_size, axis=-1)
    feed_dict = {}
    for model in models:
        feed_dict[model.inputs.x] = x_repeat
        feed_dict[model.inputs.x_mask] = x_mask_repeat
    if graph is None:
        graph = BeamSearchGraph(models, beam_size)
    ys, parents, costs = session.run(graph.outputs, feed_dict=feed_dict)
    beams = []
    for beam in _reconstruct_hypotheses(ys, parents, costs, beam_size):
        if normalization_alpha > 0.0:
            beam = [normalize(sent, cost) for (sent, cost) in beam]
        beams.append(sorted(beam, key=lambda sent_cost: sent_cost[1]))
    return beams


def _reconstruct_hypotheses(ys, parents, cost, beam_size):
    """Converts raw beam search outputs into a more usable form.

    Args:
        ys: NumPy array with shape (max_seq_len, beam_size*batch_size).
        parents: NumPy array with same shape as ys.
        cost: NumPy array with same shape as ys.
        beam_size: integer.

    Returns:
        A list of lists of (translation, score) pairs. The outer list contains
        one list for each input sentence in the batch. The inner lists contain
        k elements (where k is the beam size).
    """
    def reconstruct_single(ys, parents, hypoId, hypo, pos):
        if pos < 0:
            hypo.reverse()
            return hypo
        else:
            hypo.append(ys[pos, hypoId])
            hypoId = parents[pos, hypoId]
            return reconstruct_single(ys, parents, hypoId, hypo, pos - 1)

    hypotheses = []
    batch_size = ys.shape[1] // beam_size
    pos = ys.shape[0] - 1
    for batch in range(batch_size):
        hypotheses.append([])
        for beam in range(beam_size):
            i = batch*beam_size + beam
            hypo = reconstruct_single(ys, parents, i, [], pos)
            hypo = numpy.trim_zeros(hypo, trim='b') # b for back
            hypo.append(0)
            hypotheses[batch].append((hypo, cost[i]))
    return hypotheses


"""Builds a graph fragment for sampling over a RNNModel."""
class SampleGraph(object):
    def __init__(self, model):
        self._sampled_ys = construct_sampling_ops(model)

    @property
    def outputs(self):
        return (self._sampled_ys)


"""Builds a graph fragment for beam search over one or more RNNModels."""
class BeamSearchGraph(object):
    def __init__(self, models, beam_size, normalization_alpha):
        self._beam_size = beam_size
        self._normalization_alpha = normalization_alpha
        self._sampled_ys, self._parents, self._cost = \
            construct_beam_search_ops(models, beam_size)

    @property
    def outputs(self):
        return (self._sampled_ys, self._parents, self._cost)

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def normalization_alpha(self):
        return self._normalization_alpha


def construct_sampling_ops(model):
    """Builds a graph fragment for sampling over a RNNModel.

    Args:
        model: a RNNModel.

    Returns:
        A Tensor with shape (max_seq_len, batch_size) containing one sampled
        translation for each input sentence in model.inputs.x.
    """
    decoder = model.decoder
    batch_size = tf.shape(decoder.init_state)[0]
    high_depth = 0 if decoder.high_gru_stack == None \
                   else len(decoder.high_gru_stack.grus)
    i = tf.constant(0)
    init_y = -tf.ones(dtype=tf.int32, shape=[batch_size])
    init_emb = tf.zeros(dtype=tf.float32,
                        shape=[batch_size,decoder.embedding_size])
    y_array = tf.TensorArray(
        dtype=tf.int32,
        size=decoder.translation_maxlen,
        clear_after_read=True, #TODO: does this help? or will it only introduce bugs in the future?
        name='y_sampled_array')
    init_loop_vars = [i, decoder.init_state, [decoder.init_state] * high_depth,
                      init_y, init_emb, y_array]

    def cond(i, base_state, high_states, prev_y, prev_emb, y_array):
        return tf.logical_and(
            tf.less(i, decoder.translation_maxlen),
            tf.reduce_any(tf.not_equal(prev_y, 0)))

    def body(i, prev_base_state, prev_high_states, prev_y, prev_emb,
             y_array):
        state1 = decoder.grustep1.forward(prev_base_state, prev_emb)
        att_ctx, att_alphas = decoder.attstep.forward(state1)
        base_state = decoder.grustep2.forward(state1, att_ctx)
        if decoder.high_gru_stack == None:
            output = base_state
            high_states = []
        else:
            if decoder.high_gru_stack.context_state_size == 0:
                output, high_states = decoder.high_gru_stack.forward_single(
                    prev_high_states, base_state)
            else:
                output, high_states = decoder.high_gru_stack.forward_single(
                    prev_high_states, base_state, context=att_ctx)

        if decoder.lexical_layer is not None:
            lexical_state = decoder.lexical_layer.forward(decoder.x_embs, att_alphas)
        else:
           lexical_state = None

        logits = decoder.predictor.get_logits(prev_emb, output, att_ctx, lexical_state,
                                           multi_step=False)
        logits = model.sampling_utils.adjust_logits(logits)
        new_y = tf.multinomial(logits, num_samples=1)
        new_y = tf.cast(new_y, dtype=tf.int32)
        new_y = tf.squeeze(new_y, axis=1)
        new_y = tf.where(tf.equal(prev_y, tf.constant(0, dtype=tf.int32)),
                         tf.zeros_like(new_y), new_y)
        y_array = y_array.write(index=i, value=new_y)
        new_emb = decoder.y_emb_layer.forward(new_y, factor=0)
        return i+1, base_state, high_states, new_y, new_emb, y_array

    final_loop_vars = tf.while_loop(
                       cond=cond,
                       body=body,
                       loop_vars=init_loop_vars,
                       back_prop=False)
    i, _, _, _, _, y_array = final_loop_vars
    sampled_ys = y_array.gather(tf.range(0, i))
    return sampled_ys


def construct_beam_search_ops(models, beam_size):
    """Builds a graph fragment for beam search over one or more RNNModels.

    Strategy:
        compute the log_probs - same as with sampling
        for sentences that are ended set log_prob(<eos>)=0, log_prob(not eos)=-inf
        add previous cost to log_probs
        run top k -> (idxs, values)
        use values as new costs
        divide idxs by num_classes to get state_idxs
        use gather to get new states
        take the remainder of idxs after num_classes to get new_predicted words
    """

    # Get some parameter settings.  For ensembling, some parameters are required
    # to be consistent across all models but others are not.  In the former
    # case, we assume that consistency has already been checked.  For the
    # parameters that are allowed to vary across models, the first model's
    # settings take precedence.
    decoder = models[0].decoder
    batch_size = tf.shape(decoder.init_state)[0]
    embedding_size = decoder.embedding_size
    translation_maxlen = decoder.translation_maxlen
    target_vocab_size = decoder.target_vocab_size
    high_depth = 0 if decoder.high_gru_stack == None \
                   else len(decoder.high_gru_stack.grus)

    # Initialize loop variables
    i = tf.constant(0)
    init_ys = -tf.ones(dtype=tf.int32, shape=[batch_size])
    init_embs = [tf.zeros(dtype=tf.float32, shape=[batch_size,embedding_size])] * len(models)

    f_min = numpy.finfo(numpy.float32).min
    init_cost = [0.] + [f_min]*(beam_size-1) # to force first top k are from first hypo only
    init_cost = tf.constant(init_cost, dtype=tf.float32)
    init_cost = tf.tile(init_cost, multiples=[batch_size//beam_size])
    ys_array = tf.TensorArray(
                dtype=tf.int32,
                size=translation_maxlen,
                clear_after_read=True,
                name='y_sampled_array')
    p_array = tf.TensorArray(
                dtype=tf.int32,
                size=translation_maxlen,
                clear_after_read=True,
                name='parent_idx_array')
    init_base_states = [m.decoder.init_state for m in models]
    init_high_states = [[m.decoder.init_state] * high_depth for m in models]
    init_loop_vars = [i, init_base_states, init_high_states, init_ys, init_embs,
                      init_cost, ys_array, p_array]

    # Prepare cost matrix for completed sentences -> Prob(EOS) = 1 and Prob(x) = 0
    eos_log_probs = tf.constant(
                        [[0.] + ([f_min]*(target_vocab_size - 1))],
                        dtype=tf.float32)
    eos_log_probs = tf.tile(eos_log_probs, multiples=[batch_size,1])

    def cond(i, prev_base_states, prev_high_states, prev_ys, prev_embs, cost, ys_array, p_array):
        return tf.logical_and(
                tf.less(i, translation_maxlen),
                tf.reduce_any(tf.not_equal(prev_ys, 0)))

    def body(i, prev_base_states, prev_high_states, prev_ys, prev_embs, cost, ys_array, p_array):
        # get predictions from all models and sum the log probs
        sum_log_probs = None
        base_states = [None] * len(models)
        high_states = [None] * len(models)
        for j in range(len(models)):
            d = models[j].decoder
            states1 = d.grustep1.forward(prev_base_states[j], prev_embs[j])
            att_ctx, att_alphas = d.attstep.forward(states1)
            base_states[j] = d.grustep2.forward(states1, att_ctx)
            if d.high_gru_stack == None:
                stack_output = base_states[j]
                high_states[j] = []
            else:
                if d.high_gru_stack.context_state_size == 0:
                    stack_output, high_states[j] = d.high_gru_stack.forward_single(
                        prev_high_states[j], base_states[j])
                else:
                    stack_output, high_states[j] = d.high_gru_stack.forward_single(
                        prev_high_states[j], base_states[j], context=att_ctx)

            if d.lexical_layer is not None:
                lexical_state = d.lexical_layer.forward(d.x_embs, att_alphas)
            else:
                lexical_state = None

            logits = d.predictor.get_logits(prev_embs[j], stack_output,
                                            att_ctx, lexical_state, multi_step=False)
            log_probs = tf.nn.log_softmax(logits) # shape (batch, vocab_size)
            if sum_log_probs == None:
                sum_log_probs = log_probs
            else:
                sum_log_probs += log_probs

        # set cost of EOS to zero for completed sentences so that they are in top k
        # Need to make sure only EOS is selected because a completed sentence might
        # kill ongoing sentences
        sum_log_probs = tf.where(tf.equal(prev_ys, 0), eos_log_probs, sum_log_probs)

        all_costs = sum_log_probs + tf.expand_dims(cost, axis=1) # TODO: you might be getting NaNs here since -inf is in log_probs

        all_costs = tf.reshape(all_costs,
                               shape=[-1, target_vocab_size * beam_size])
        values, indices = tf.nn.top_k(all_costs, k=beam_size) #the sorted option is by default True, is this needed?
        new_cost = tf.reshape(values, shape=[batch_size])
        offsets = tf.range(
                    start = 0,
                    delta = beam_size,
                    limit = batch_size,
                    dtype=tf.int32)
        offsets = tf.expand_dims(offsets, axis=1)
        survivor_idxs = (indices // target_vocab_size) + offsets
        new_ys = indices % target_vocab_size
        survivor_idxs = tf.reshape(survivor_idxs, shape=[batch_size])
        new_ys = tf.reshape(new_ys, shape=[batch_size])
        new_embs = [m.decoder.y_emb_layer.forward(new_ys, factor=0) for m in models]
        new_base_states = [tf.gather(s, indices=survivor_idxs) for s in base_states]
        new_high_states = [[tf.gather(s, indices=survivor_idxs) for s in states] for states in high_states]
        new_cost = tf.where(tf.equal(new_ys, 0), tf.abs(new_cost), new_cost)

        ys_array = ys_array.write(i, value=new_ys)
        p_array = p_array.write(i, value=survivor_idxs)

        return i+1, new_base_states, new_high_states, new_ys, new_embs, new_cost, ys_array, p_array


    final_loop_vars = tf.while_loop(
                        cond=cond,
                        body=body,
                        loop_vars=init_loop_vars,
                        back_prop=False)
    i, _, _, _, _, cost, ys_array, p_array = final_loop_vars

    indices = tf.range(0, i)
    sampled_ys = ys_array.gather(indices)
    parents = p_array.gather(indices)
    cost = tf.abs(cost) #to get negative-log-likelihood
    return sampled_ys, parents, cost
