import tensorflow as tf

from transformer_layers import get_shape_list
import tf_utils

class ModelAdapter:
    """Implements model-specific functionality needed by the *Sampler classes.

    The BeamSearchSampler and RandomSampler classes need to work with RNN and
    Transformer models, which have different interfaces (and obviously
    different architectures). This class hides the RNN-specific details behind
    a common interace (see transformer_inference.ModelAdapter for the
    Transformer counterpart).
    """
    def __init__(self, model, config, scope):
        self._model = model
        self._config = config
        self._scope = scope

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._config

    @property
    def target_vocab_size(self):
        return self._model.decoder.target_vocab_size

    @property
    def batch_size(self):
        return tf.shape(self._model.inputs.x)[-1]

    def encode(self):
        return None

    def generate_decoding_function(self, encoder_output):

        def _decoding_function_outer(step_target_ids, current_time_step,
                                     memories):
            """Single-step decoding function (outer version).

            This is a wrapper around _decoding_function_inner() that does some
            housekeeping before calling that function to do the actual work.

            Args:
                step_target_ids: Tensor with shape (batch_size)
                current_time_step: scalar Tensor.
                memories: dictionary (see top-level class description)

            Returns:
            """
            with tf.name_scope(self._scope):

                shapes = { step_target_ids: ('batch_size',) }
                tf_utils.assert_shapes(shapes)

                logits, memories['base_states'], memories['high_states'] = \
                    _decoding_function_inner(
                        step_target_ids, memories['base_states'],
                        memories['high_states'], current_time_step)

                return logits, memories

        def _decoding_function_inner(vocab_ids, prev_base_states,
                                     prev_high_states, current_time_step):
            """Single-step decoding function (inner version).

            Args:
                vocab_ids: TODO
                prev_base_states: TODO
                prev_high_states: TODO
                current_time_step: TODO

            Returns:
            """
            d = self._model.decoder

            # The first time step is a special case for the RNN since there is
            # no (valid) input token and no word embeddings to lookup. Like in
            # training, we use zero-valued dummy embeddings.
            # This differs from the Transformer model, which has a BOS token
            # (called <GO>) with an associated embedding that is learned during
            # training.
            embeddings = tf.cond(
                tf.equal(current_time_step, 1),
                lambda: d.y_emb_layer.zero(vocab_ids, factor=0),
                lambda: d.y_emb_layer.forward(vocab_ids, factor=0))

            states1 = d.grustep1.forward(prev_base_states, embeddings)
            att_ctx, att_alphas = d.attstep.forward(states1)
            base_states = d.grustep2.forward(states1, att_ctx)

            if d.high_gru_stack is None:
                stack_output = base_states
                high_states = []
            elif d.high_gru_stack.context_state_size == 0:
                stack_output, high_states = d.high_gru_stack.forward_single(
                    prev_high_states, base_states)
            else:
                stack_output, high_states = d.high_gru_stack.forward_single(
                    prev_high_states, base_states, context=att_ctx)

            if d.lexical_layer is not None:
                lexical_state = d.lexical_layer.forward(d.x_embs, att_alphas)
            else:
                lexical_state = None

            logits = d.predictor.get_logits(
                embeddings, stack_output, att_ctx, lexical_state,
                multi_step=False)

            return logits, base_states, high_states

        return _decoding_function_outer

    def generate_initial_memories(self, batch_size, beam_size):
        with tf.name_scope(self._scope):
            d = self._model.decoder

            shapes = { d.init_state: ('batch_size', self.config.state_size) }
            tf_utils.assert_shapes(shapes)

            high_depth = 0 if d.high_gru_stack is None \
                           else len(d.high_gru_stack.grus)

            initial_memories = {}
            initial_memories['base_states'] = d.init_state
            initial_memories['high_states'] = [d.init_state] * high_depth
            return initial_memories


    def get_memory_invariants(self, memories):
        """Generate shape invariants for memories.

        Args:
            memories: dictionary (see top-level class description)

        Returns:
            Dictionary of shape invariants with same structure as memories.
        """
        with tf.name_scope(self._scope):
            d = self._model.decoder

            high_depth = 0 if d.high_gru_stack is None \
                           else len(d.high_gru_stack.grus)

            num_dims = len(get_shape_list(memories['base_states']))
            # TODO Specify shape in full?
            partial_shape = tf.TensorShape([None] * num_dims)

            invariants = {}
            invariants['base_states'] = partial_shape
            invariants['high_states'] = [partial_shape] * high_depth
            return invariants


    def gather_memories(self, memories, gather_coordinates):
        """Gathers memories for selected beam entries.

        Args:
            memories: dictionary (see top-level class description)
            gather_coordinates: Tensor with shape [batch_size_x, beam_size, 2]

        Returns:
            Dictionary containing gathered memories.
        """
        with tf.name_scope(self._scope):

            shapes = { gather_coordinates: ('batch_size_x', 'beam_size', 2) }
            tf_utils.assert_shapes(shapes)

            coords_shape = tf.shape(gather_coordinates)
            batch_size_x, beam_size = coords_shape[0], coords_shape[1]

            def gather_states(states):
                shapes = { states: ('batch_size', self._config.state_size) }
                tf_utils.assert_shapes(shapes)
                states_shape = tf.shape(states)
                state_size = states_shape[1]
                tmp = tf.reshape(states, [beam_size, batch_size_x, state_size])
                flat_tensor = tf.transpose(tmp, [1, 0, 2])
                tmp = tf.gather_nd(flat_tensor, gather_coordinates)
                tmp = tf.transpose(tmp, [1, 0, 2])
                gathered_values = tf.reshape(tmp, states_shape)
                return gathered_values

            gathered_memories = {}

            gathered_memories['base_states'] = \
                gather_states(memories['base_states'])

            gathered_memories['high_states'] = [
                gather_states(states) for states in memories['high_states']
            ]

            return gathered_memories
