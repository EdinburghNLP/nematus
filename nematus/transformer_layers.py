"""Adapted from Nematode: https://github.com/demelin/nematode """

# TODO: Add an attention visualization component - very important (~easy)

""" Layer implementations. """

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer


def matmul_nd(nd_tensor, matrix):
    """ Performs matrix multiplication for n-dimensional inputs. """
    tensor_shape = get_shape_list(nd_tensor)
    matrix_shape = get_shape_list(matrix)

    initial_tensor_dims = tensor_shape[:-1]
    flat_first_dim = tf.reduce_prod(initial_tensor_dims)

    tensor_2d = tf.reshape(nd_tensor, [flat_first_dim, tensor_shape[-1]])
    result_2d = tf.matmul(tensor_2d, matrix)
    result_3d = tf.reshape(result_2d, initial_tensor_dims + [matrix_shape[-1]])
    return result_3d


def get_shape_list(inputs):
    """ Returns a list of input dimensions, statically where possible; adopted from the tensor2tensor library. """
    inputs = tf.convert_to_tensor(inputs)
    # If inputs rank is unknown, return dynamic shape
    if inputs.get_shape().dims is None:
        dims_list = tf.shape(inputs)
    else:
        static_dims = inputs.get_shape().as_list()
        shape = tf.shape(inputs)
        # Filter out non-specified dimensions and replace them with static shape definitions
        dims_list = list()
        for i in range(len(static_dims)):
            dim = static_dims[i]
            if dim is None:
                dim = shape[i]
            dims_list.append(dim)
    return dims_list


def get_right_context_mask(time_steps):
    """ Generates the mask preventing the decoder from attending to unseen positions. """
    # Generate mask that limits decoder self-attention up to and including the current position
    attn_mask = tf.matrix_band_part(tf.ones([time_steps, time_steps]), -1, 0)
    # Expand mask to 4d. so as to be compatible with attention weights
    attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 0), 0)
    # Illegal connections will be set to -inf when fed into the softmax function
    # Padding for non-masked positions is applied to prevent NaNs
    attn_mask = -1e9 * (1.0 - attn_mask)
    return attn_mask


def get_positional_signal(time_steps, depth, float_dtype, min_timescale=1, max_timescale=10000):
    """ Generates a series of sinusoid functions capable of expressing the relative and absolute position
    of a token within a longer sequence. """
    # Convert to floats
    min_timescale = tf.cast(min_timescale, float_dtype)
    max_timescale = tf.cast(max_timescale, float_dtype)
    # Obtain timing signal via sinusoids
    num_timescales = tf.cast(depth // 2, float_dtype)
    log_timescale_increment = tf.log(max_timescale / min_timescale) / (num_timescales - tf.cast(1.0, float_dtype))
    # Introduce an offset between individual timescales to obtain different frequencies
    incremented_timescales = \
        min_timescale * tf.exp(tf.range(num_timescales, dtype=float_dtype) * -log_timescale_increment)
    # Assign the designated number of time-scales per token position
    positions = tf.cast(tf.range(time_steps), float_dtype)
    scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(incremented_timescales, 0)
    positional_signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    # Pad the signal tensor, if needed
    pad_size = depth % 2
    if pad_size != 0:
        tf.pad(positional_signal, [[0, 0], [0, pad_size]])
    # Reshape the signal to make it compatible with the target tensor
    positional_signal = tf.reshape(positional_signal, [1, time_steps, depth])
    return positional_signal


class EmbeddingLayer(object):
    """ Looks up embeddings for the specified token sequence in the learned embedding table; allows for easy weight
    scaling and tying. """

    def __init__(self, vocabulary_size, embedding_size, hidden_size, float_dtype, name):
        # Set arguments
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.float_dtype = float_dtype
        self.name = name

        # Create embedding matrix and its transposes
        with tf.variable_scope(self.name):
            self.embedding_table = tf.get_variable(name='embedding_table',
                                                shape=[vocabulary_size, embedding_size],
                                                dtype=float_dtype,
                                                initializer=glorot_uniform_initializer(),
                                                trainable=True)
            self.projection_matrix = tf.transpose(self.embedding_table, name='vocab_projection_matrix')

    def embed(self, one_hot_inputs):
        """ Embeds one-hot-vectors corresponding to input tokens. """
        embeddings = tf.nn.embedding_lookup(self.embedding_table, one_hot_inputs)
        # Apply transformer-specific scaling
        embeddings *= tf.sqrt(tf.cast(self.hidden_size, self.float_dtype))
        return embeddings

    def project(self, dec_out):
        """ Projects the transformer decoder's output into the vocabulary space. """
        projections = matmul_nd(dec_out, self.projection_matrix)
        return projections

    def get_embedding_table(self):
        """ Recovers the learned embedding table. """
        return self.embedding_table

    def get_projection_matrix(self):
        """ Recovers the pre-softmax projection matrix which is the inverse of the embedding table. """
        return self.projection_matrix

    def get_vocab_size(self):
        """ Recovers the vocabulary size. """
        return self.vocabulary_size


class LayerNormLayer(object):
    """ Performs layer normalization by computing the mean and variance used for normalization from all of the
    summed inputs to neurons in a layer. """

    def __init__(self, dims_out, name=None, eps=1e-5):
        if name is None:
            name = 'layer_norm'
        else:
            name = '{:s}_layer_norm'.format(name)

        with tf.variable_scope(name, values=[dims_out]):
            self.offset = tf.get_variable(name='offset',
                                          shape=[dims_out],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            self.scale = tf.get_variable(name='scale',
                                         shape=[dims_out],
                                         dtype=tf.float32,
                                         initializer=tf.ones_initializer())
            self.eps = tf.constant(eps)

    def forward(self, inputs):
        layer_mean, layer_var = tf.nn.moments(inputs, axes=-1, keep_dims=True)
        normalized = tf.add(
            tf.multiply(self.scale, tf.math.divide(tf.subtract(inputs, layer_mean),
                                           tf.sqrt(tf.add(layer_var, self.eps)))),
            self.offset)

        return normalized


class ProcessingLayer(object):
    """ Optionally applies residual connections, layer normalization, or dropout. """

    def __init__(self, out_size, use_layer_norm, dropout_rate, training, name):
        # Set attributes
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.training = training
        self.name = name

        # Initialize layer normalization, if specified
        with tf.variable_scope(self.name):
            if use_layer_norm:
                self.layer_norm = LayerNormLayer(out_size)

    def forward(self, inputs, residual_inputs=None):
        with tf.variable_scope(self.name, values=[inputs, residual_inputs], reuse=True):
            outputs = inputs
            # Apply dropout
            if self.dropout_rate > 0.0:
                outputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self.training)
            # Apply residual connections
            if residual_inputs is not None:
                outputs = outputs + residual_inputs
            # Apply layer normalization
            if self.use_layer_norm:
                outputs = self.layer_norm.forward(outputs)
        return outputs


class FeedForwardLayer(object):
    """ A single fully-connected feed-forward layer using standard dropout. """

    def __init__(self,
                 in_size,
                 out_size,
                 float_dtype,
                 dropout_rate,
                 activation,
                 use_bias,
                 use_layer_norm,
                 training,
                 name):
        # Set attributes
        self.in_size = in_size
        self.out_size = out_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.training = training
        self.name = name

        with tf.variable_scope(self.name):
            # Set up layer normalization
            if use_layer_norm:
                self.layer_norm_layer = LayerNormLayer(out_size)
            else:
                self.layer_norm_layer = None

            # Define parameters
            weights_shape = [in_size, out_size] if out_size is not None else [in_size]
            self.weights = tf.get_variable(name='dense_layer_weights',
                                           shape=weights_shape,
                                           dtype=float_dtype,
                                           initializer=glorot_uniform_initializer(),
                                           trainable=True)
            if use_bias:
                biases_shape = [out_size] if out_size is not None else [in_size]
                self.biases = tf.get_variable(name='dense_layer_biases',
                                              shape=biases_shape,
                                              dtype=float_dtype,
                                              initializer=tf.zeros_initializer(),
                                              trainable=True)

    def forward(self, inputs):
        with tf.variable_scope(self.name, values=[inputs]):
            # Optionally apply dropout
            if self.dropout_rate > 0.0:
                inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self.training)
            # Feed through a dense layer
            outputs = matmul_nd(inputs, self.weights)
            if self.use_bias:
                outputs += self.biases
            if self.activation is not None:
                outputs = self.activation(outputs)
            # Optionally apply layer normalization
            if self.layer_norm_layer is not None:
                outputs = self.layer_norm_layer(outputs)
            return outputs


class FeedForwardNetwork(object):
    """ A fully connected feed-forward network that is applied to each position separately and identically. """

    def __init__(self,
                 layer_dims,
                 float_dtype,
                 use_bias,
                 activation,
                 use_layer_norm,
                 dropout_rate,
                 training,
                 name=None):
        # Set attributes
        self.layer_dims = layer_dims
        self.float_dtype = float_dtype
        self.use_bias = use_bias
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.training = training
        self.name = name
        # Container for network layers
        self.layers = list()
        self._initialize_layers()

    def _initialize_layers(self):
        """ Builds the network from fully-connected layers. """
        num_layers = len(self.layer_dims)
        for layer_id in range(num_layers):
            # Assure that no non-linearity or dropout is applied at the final layer
            if layer_id == num_layers - 1:
                layer_activation = None
                dropout_rate = 0.0
            else:
                layer_activation = self.activation
                dropout_rate = self.dropout_rate
            # Add layer
            if layer_id == 0:
                input_dims = self.layer_dims[-1]  # input and output dimensions of the sub-layer are identical
            else:
                input_dims = self.layer_dims[layer_id - 1]
            self.layers.append(FeedForwardLayer(input_dims,
                                                self.layer_dims[layer_id],
                                                self.float_dtype,
                                                dropout_rate=dropout_rate,
                                                activation=layer_activation,
                                                use_bias=self.use_bias,
                                                use_layer_norm=self.use_layer_norm,
                                                training=self.training,
                                                name='ff_layer_{:d}'.format(layer_id + 1)))

    def forward(self, inputs):
        """ Propagates input data through the specified layers. """
        with tf.variable_scope(self.name, values=[inputs]):
            for layer in self.layers:
                inputs = layer.forward(inputs)
            return inputs


class PReLU(object):
    """ Implements the adaptive Parametric Rectified Linear Unit activation function. """

    def __init__(self,
                 in_size,
                 initial_slope=1.0,
                 name=None):
        with tf.variable_scope(name, default_name='PReLu'):
            self.slope = tf.Variable(initial_slope * np.ones((in_size,)).astype('float32'), name='slope')

    def forward(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = inputs - pos
        outputs = pos + self.slope * neg
        return outputs


class MaskedCrossEntropy(object):
    """ Implements the cross-entropy loss with optionally applied label smoothing for better model generalization. """
    def __init__(self, vocab_size, label_smoothing_discount, int_dtype, float_dtype, time_major, name=None):
        # Set attributes
        self.vocab_size = vocab_size
        self.label_smoothing_discount = label_smoothing_discount
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.time_dim = int(not time_major)  # i.e. 0 is time_major, 1 if batch_major
        self.name = name

    def _get_smoothing_parameters(self):
        """ Calculates the confidence values used for label smoothing application. """
        # Assign low confidence, i.e. the label smoothing discount value, to all non-true labels
        one_out_vocab = tf.cast(self.vocab_size - 1, self.float_dtype)
        # For cross-entropy, each row of the labels matrix must be a valid probability distribution
        low_confidence = self.label_smoothing_discount / one_out_vocab
        high_confidence = 1.0 - self.label_smoothing_discount
        # Normalizing constant for better readability, which is the best cross-entropy value with soft targets
        # Has no impact on training
        normalizing_factor = -(1.0 * high_confidence * tf.log(high_confidence)
                               + one_out_vocab * low_confidence * tf.log(low_confidence + 1e-20))
        return high_confidence, low_confidence, normalizing_factor

    def forward(self, logits, targets, target_mask, training):
        with tf.name_scope(self.name, values=[logits, targets, target_mask]):
            # Get smoothing parameters (no smoothing/ normalization at test time)
            high_confidence, low_confidence, normalizing_factor = \
                tf.cond(tf.logical_and(training, tf.greater(self.label_smoothing_discount, 0.0)),
                        self._get_smoothing_parameters,
                        lambda: (1.0, 0.0, 0.0))

            # If necessary, pad the label and the label-mask to match the length of decoder output
            # Not sure if that's a sensible thing to do
            targets_shape = tf.shape(targets)
            logits_shape = tf.shape(logits)
            targets_length = targets_shape[self.time_dim]
            logits_length = logits_shape[self.time_dim]

            def _get_pad_shape(shape_to_pad, shape_to_match):
                """ Calculates the shape of the padding to be applied to the logits or targets. """
                time_steps_to_pad = shape_to_match[self.time_dim] - shape_to_pad[self.time_dim]
                if self.time_dim == 0:
                    pad_shape = [time_steps_to_pad, shape_to_pad[1]]
                else:
                    pad_shape = [shape_to_pad[0], time_steps_to_pad]
                return pad_shape

            def _pad_targets(targets, target_mask, logits):
                """ Pads the targets to match the size of the model-generated logits. """
                pad_shape = _get_pad_shape(targets_shape, logits_shape)
                targets = tf.concat([targets, tf.zeros(pad_shape, dtype=self.int_dtype)], axis=self.time_dim)
                target_mask = tf.concat([target_mask, tf.zeros(pad_shape, dtype=self.float_dtype)], axis=self.time_dim)
                return targets, target_mask, logits

            def _pad_logits(targets, target_mask, logits):
                """ Pads the logits to match the size of the ground-truth targets. """
                pad_shape = _get_pad_shape(logits_shape, targets_shape)
                logits = tf.concat([logits, tf.zeros(pad_shape + [logits_shape[-1]], dtype=self.float_dtype)],
                                   axis=self.time_dim)
                return targets, target_mask, logits

            # For teacher-forcing with RNN models
            targets, target_mask, logits = tf.cond(tf.equal(targets_length, logits_length),
                                                   lambda: (targets, target_mask, logits),
                                                   lambda: tf.cond(tf.less(targets_length, logits_length),
                                                                   lambda: _pad_targets(targets, target_mask, logits),
                                                                   lambda: _pad_logits(targets, target_mask, logits)))

            # Project and optionally smooth target token ids
            projected_targets = tf.one_hot(targets,
                                           depth=self.vocab_size,
                                           on_value=high_confidence,
                                           off_value=low_confidence,
                                           dtype=self.float_dtype)

            # Compute token-level loss
            flat_logits = tf.reshape(logits, [-1, self.vocab_size])
            flat_targets = tf.reshape(projected_targets, [-1, self.vocab_size])
            flat_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_targets)
            flat_normalized_loss = flat_loss - normalizing_factor
            # Compute sentence- and batch-level losses (i.e. mean token-loss per sentence/ batch)
            normalized_loss = tf.reshape(flat_normalized_loss, tf.shape(targets))
            masked_loss = normalized_loss * target_mask
            sentence_lengths = tf.reduce_sum(target_mask, axis=self.time_dim, keepdims=False)
            sentence_loss = tf.math.divide(tf.reduce_sum(masked_loss, axis=self.time_dim, keepdims=False), sentence_lengths)
            batch_loss = tf.reduce_mean(sentence_loss, keepdims=False)
        return masked_loss, sentence_loss, batch_loss
