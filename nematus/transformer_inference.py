"""Adapted from Nematode: https://github.com/demelin/nematode """

import sys
import tensorflow as tf

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import tf_utils
    from .transformer import INT_DTYPE, FLOAT_DTYPE
    from .transformer_layers import get_positional_signal
except (ModuleNotFoundError, ImportError) as e:
    import tf_utils
    from transformer import INT_DTYPE, FLOAT_DTYPE
    from transformer_layers import get_positional_signal


class EncoderOutput:
    def __init__(self, enc_output, cross_attn_mask):
        self.enc_output = enc_output
        self.cross_attn_mask = cross_attn_mask


class ModelAdapter:
    """Implements model-specific functionality needed by the *Sampler classes.

    The BeamSearchSampler and RandomSampler classes need to work with RNN and
    Transformer models, which have different interfaces (and obviously
    different architectures). This class hides the Transformer-specific details
    behind a common interace (see rnn_inference.ModelAdapter for the RNN
    counterpart).
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
        return self._model.dec.embedding_layer.get_vocab_size()

    @property
    def batch_size(self):
        return tf.shape(input=self._model.inputs.x)[-1]

    def encode(self):
        with tf.compat.v1.name_scope(self._scope):
            enc_output, cross_attn_mask = self._model.enc.encode(
                self._model.source_ids, self._model.source_mask)
            return EncoderOutput(enc_output, cross_attn_mask)

    def generate_decoding_function(self, encoder_output):

        with tf.compat.v1.name_scope(self._scope):
            # Generate a positional signal for the longest possible output.
            positional_signal = get_positional_signal(
                self._config.translation_maxlen,
                self._config.embedding_size,
                FLOAT_DTYPE)

        decoder = self._model.dec

        if self.config.transformer_dropout_embeddings > 0:
            dropout = tf.keras.layers.Dropout(rate=self.config.transformer_dropout_embeddings)
        else:
            dropout = None

        def _decoding_function(step_target_ids, current_time_step, memories):
            """Single-step decoding function.

            Args:
                step_target_ids: Tensor with shape (batch_size)
                current_time_step: scalar Tensor.
                memories: dictionary (see top-level class description)

            Returns:
            """
            with tf.compat.v1.name_scope(self._scope):
                # TODO Is this necessary?
                vocab_ids = tf.reshape(step_target_ids, [-1, 1])
                # Look up embeddings for target IDs.
                target_embeddings = decoder._embed(vocab_ids)
                # Add positional signal.
                signal_slice = positional_signal[
                    :, current_time_step-1:current_time_step, :]
                target_embeddings += signal_slice
                # Optionally, apply dropout to embeddings.
                if dropout is not None:
                    target_embeddings = dropout(
                        target_embeddings,
                        training=decoder.training)
                # Propagate values through the decoder stack.
                # NOTE: No self-attention mask is applied at decoding, as
                #       future information is unavailable.
                layer_output = target_embeddings
                for layer_id in range(1, self.config.transformer_dec_depth+1):
                    layer = decoder.decoder_stack[layer_id]
                    mem_key = 'layer_{:d}'.format(layer_id)
                    layer_output, memories[mem_key] = \
                        layer['self_attn'].forward(
                            layer_output, None, None, memories[mem_key])
                    layer_output, _ = layer['cross_attn'].forward(
                        layer_output, encoder_output.enc_output,
                        encoder_output.cross_attn_mask)
                    layer_output = layer['ffn'].forward(layer_output)
                # Return prediction at the final time-step to be consistent
                # with the inference pipeline.
                dec_output = layer_output[:, -1, :]
                # Project decoder stack outputs and apply the soft-max
                # non-linearity.
                step_logits = \
                    decoder.softmax_projection_layer.project(dec_output)
                return step_logits, memories

        return _decoding_function

    def generate_initial_memories(self, batch_size, beam_size):
        with tf.compat.v1.name_scope(self._scope):
            state_size = self.config.state_size
            memories = {}
            for layer_id in range(1, self.config.transformer_dec_depth + 1):
                memories['layer_{:d}'.format(layer_id)] = { \
                    'keys': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                    [beam_size, 1, 1]),
                    'values': tf.tile(tf.zeros([batch_size, 0, state_size]),
                                      [beam_size, 1, 1])
                }
            return memories

    def get_memory_invariants(self, memories):
        """Generate shape invariants for memories.

        Args:
            memories: dictionary (see top-level class description)

        Returns:
            Dictionary of shape invariants with same structure as memories.
        """
        with tf.compat.v1.name_scope(self._scope):
            invariants = dict()
            for layer_id in memories.keys():
                layer_mems = memories[layer_id]
                invariants[layer_id] = {
                    key: tf.TensorShape(
                        [None]*len(tf_utils.get_shape_list(layer_mems[key])))
                    for key in layer_mems.keys()
                }
            return invariants

    def gather_memories(self, memories, gather_coordinates):
        """ Gathers layer-wise memory tensors for selected beam entries.

        Args:
            memories: dictionary (see top-level class description)
            gather_coordinates: Tensor with shape [batch_size_x, beam_size, 2]

        Returns:
            Dictionary containing gathered memories.
        """
        with tf.compat.v1.name_scope(self._scope):

            shapes = { gather_coordinates: ('batch_size_x', 'beam_size', 2) }
            tf_utils.assert_shapes(shapes)

            coords_shape = tf.shape(input=gather_coordinates)
            batch_size_x, beam_size = coords_shape[0], coords_shape[1]

            def gather_attn(attn):
                # TODO Specify second and third?
                shapes = { attn: ('batch_size', None, None) }
                tf_utils.assert_shapes(shapes)
                attn_dims = tf_utils.get_shape_list(attn)
                new_shape = [beam_size, batch_size_x] + attn_dims[1:]
                tmp = tf.reshape(attn, new_shape)
                flat_tensor = tf.transpose(a=tmp, perm=[1, 0, 2, 3])
                tmp = tf.gather_nd(flat_tensor, gather_coordinates)
                tmp = tf.transpose(a=tmp, perm=[1, 0, 2, 3])
                gathered_values = tf.reshape(tmp, attn_dims)
                return gathered_values

            gathered_memories = dict()

            for layer_key in memories.keys():
                layer_dict = memories[layer_key]
                gathered_memories[layer_key] = dict()

                for attn_key in layer_dict.keys():
                    attn_tensor = layer_dict[attn_key]
                    gathered_memories[layer_key][attn_key] = \
                        gather_attn(attn_tensor)

            return gathered_memories
