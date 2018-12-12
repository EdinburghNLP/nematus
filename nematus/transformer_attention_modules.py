"""Adapted from Nematode: https://github.com/demelin/nematode """

import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from transformer_layers import \
    get_shape_list, \
    FeedForwardLayer, \
    matmul_nd


class MultiHeadAttentionLayer(object):
    """ Defines the multi-head, multiplicative attention mechanism;
    based on the tensor2tensor library implementation. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 total_key_dims,
                 total_value_dims,
                 output_dims,
                 num_heads,
                 float_dtype,
                 dropout_attn,
                 training,
                 name=None):

        # Set attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.total_key_dims = total_key_dims
        self.total_value_dims = total_value_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.training = training
        self.name = name

        # Check if the specified hyper-parameters are consistent
        if total_key_dims % num_heads != 0:
            raise ValueError('Specified total attention key dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_key_dims, num_heads))
        if total_value_dims % num_heads != 0:
            raise ValueError('Specified total attention value dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_value_dims, num_heads))

        # Instantiate parameters
        with tf.variable_scope(self.name):
            self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='queries_projection')

            self.keys_projection = FeedForwardLayer(self.reference_dims,
                                                    self.total_key_dims,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

            self.values_projection = FeedForwardLayer(self.reference_dims,
                                                      self.total_value_dims,
                                                      float_dtype,
                                                      dropout_rate=0.,
                                                      activation=None,
                                                      use_bias=False,
                                                      use_layer_norm=False,
                                                      training=self.training,
                                                      name='values_projection')

            self.context_projection = FeedForwardLayer(self.total_value_dims,
                                                       self.output_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='context_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = self.values_projection.forward(memory_context)
        return queries, keys, values

    def _split_among_heads(self, inputs):
        """ Splits the attention inputs among multiple heads. """
        # Retrieve the depth of the input tensor to be split (input is 3d)
        inputs_dims = get_shape_list(inputs)
        inputs_depth = inputs_dims[-1]

        # Assert the depth is compatible with the specified number of attention heads
        if isinstance(inputs_depth, int) and isinstance(self.num_heads, int):
            assert inputs_depth % self.num_heads == 0, \
                ('Attention inputs depth {:d} is not evenly divisible by the specified number of attention heads {:d}'
                 .format(inputs_depth, self.num_heads))
        split_inputs = tf.reshape(inputs, inputs_dims[:-1] + [self.num_heads, inputs_depth // self.num_heads])
        return split_inputs

    def _merge_from_heads(self, split_inputs):
        """ Inverts the _split_among_heads operation. """
        # Transpose split_inputs to perform the merge along the last two dimensions of the split input
        split_inputs = tf.transpose(split_inputs, [0, 2, 1, 3])
        # Retrieve the depth of the tensor to be merged
        split_inputs_dims = get_shape_list(split_inputs)
        split_inputs_depth = split_inputs_dims[-1]
        # Merge the depth and num_heads dimensions of split_inputs
        merged_inputs = tf.reshape(split_inputs, split_inputs_dims[:-2] + [self.num_heads * split_inputs_depth])
        return merged_inputs

    def _dot_product_attn(self, queries, keys, values, attn_mask, scaling_on):
        """ Defines the dot-product attention function; see Vasvani et al.(2017), Eq.(1). """
        # query/ key/ value have shape = [batch_size, time_steps, num_heads, num_features]
        # Tile keys and values tensors to match the number of decoding beams; ignored if already done by fusion module
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(keys, [num_beams, 1, 1, 1]), lambda: keys)
        values = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(values, [num_beams, 1, 1, 1]), lambda: values)

        # Transpose split inputs
        queries = tf.transpose(queries, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])
        attn_logits = tf.matmul(queries, tf.transpose(keys, [0, 2, 3, 1]))

        # Scale attention_logits by key dimensions to prevent softmax saturation, if specified
        if scaling_on:
            key_dims = get_shape_list(keys)[-1]
            normalizer = tf.sqrt(tf.cast(key_dims, self.float_dtype))
            attn_logits /= normalizer

        # Optionally mask out positions which should not be attended to
        # attention mask should have shape=[batch, num_heads, query_length, key_length]
        # attn_logits has shape=[batch, num_heads, query_length, key_length]
        if attn_mask is not None:
            attn_mask = tf.cond(tf.greater(num_beams, 1),
                                lambda: tf.tile(attn_mask, [num_beams, 1, 1, 1]),
                                lambda: attn_mask)
            attn_logits += attn_mask

        # Calculate attention weights
        attn_weights = tf.nn.softmax(attn_logits)
        # Optionally apply dropout:
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Weigh attention values
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # The context for the query and the referenced memory is identical in case of self-attention
        if memory_context is None:
            memory_context = query_context

        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context)

        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Split attention inputs among attention heads
        split_queries = self._split_among_heads(queries)
        split_keys = self._split_among_heads(keys)
        split_values = self._split_among_heads(values)
        # Apply attention function
        split_weighted_memories = self._dot_product_attn(split_queries, split_keys, split_values, attn_mask,
                                                         scaling_on=True)
        # Merge head output
        weighted_memories = self._merge_from_heads(split_weighted_memories)
        # Feed through a dense layer
        projected_memories = self.context_projection.forward(weighted_memories)
        return projected_memories, layer_memories


class SingleHeadAttentionLayer(object):
    """ Single-head attention module. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 hidden_dims,
                 float_dtype,
                 dropout_attn,
                 training,
                 name,
                 attn_type='multiplicative'):

        # Declare attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.hidden_dims = hidden_dims
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.attn_type = attn_type
        self.training = training
        self.name = name

        assert attn_type in ['additive', 'multiplicative'], 'Attention type {:s} is not supported.'.format(attn_type)

        # Instantiate parameters
        with tf.variable_scope(self.name):
            self.queries_projection = None
            self.attn_weight = None
            if attn_type == 'additive':
                self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                           self.hidden_dims,
                                                           float_dtype,
                                                           dropout_rate=0.,
                                                           activation=None,
                                                           use_bias=False,
                                                           use_layer_norm=False,
                                                           training=self.training,
                                                           name='queries_projection')

                self.attn_weight = tf.get_variable(name='attention_weight',
                                                   shape=self.hidden_dims,
                                                   dtype=float_dtype,
                                                   initializer=glorot_uniform_initializer(),
                                                   trainable=True)

            self.keys_projection = FeedForwardLayer(self.reference_dims,
                                                    self.hidden_dims,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = query_context
        if self.attn_type == 'additive':
            queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = memory_context
        return queries, keys, values

    def _additive_attn(self, queries, keys, values, attn_mask):
        """ Uses additive attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return tf.reduce_sum(self.attn_weight * tf.nn.tanh(keys + query), axis=-1)

        # Obtain attention scores
        transposed_queries = tf.transpose(queries, [1, 0, 2])  # time-major
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)
        attn_logits = tf.transpose(attn_logits, [1, 0, 2])

        if attn_mask is not None:
            # Transpose and tile the mask
            attn_logits += tf.tile(tf.squeeze(attn_mask, 1),
                                   [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1])

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-1, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Obtain context vectors
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def _multiplicative_attn(self, queries, keys, values, attn_mask):
        """ Uses multiplicative attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        # Use multiplicative attention
        transposed_keys = tf.transpose(keys, [0, 2, 1])
        attn_logits = tf.matmul(queries, transposed_keys)
        if attn_mask is not None:
            # Transpose and tile the mask
            attn_logits += tf.tile(tf.squeeze(attn_mask, 1),
                                   [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1])

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-1, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Obtain context vectors
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # The context for the query and the referenced memory is identical in case of self-attention
        if memory_context is None:
            memory_context = query_context

        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context)
        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Obtain weighted layer hidden representations
        if self.attn_type == 'additive':
            weighted_memories = self._additive_attn(queries, keys, values, attn_mask)
        else:
            weighted_memories = self._multiplicative_attn(queries, keys, values, attn_mask)
        return weighted_memories, layer_memories


class FineGrainedAttentionLayer(SingleHeadAttentionLayer):
    """ Defines the fine-grained attention layer; based on
    "Fine-grained attention mechanism for neural machine translation.", Choi et al, 2018. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 hidden_dims,
                 float_dtype,
                 dropout_attn,
                 training,
                 name,
                 attn_type='multiplicative'):
        super(FineGrainedAttentionLayer, self).__init__(reference_dims, hypothesis_dims, hidden_dims, float_dtype,
                                                        dropout_attn, training, name, attn_type)

    def _additive_attn(self, queries, keys, values, attn_mask):
        """ Uses additive attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return self.attn_weight * tf.nn.tanh(keys + query)

        # Obtain attention scores
        transposed_queries = tf.transpose(queries, [1, 0, 2])  # time-major
        # attn_logits has shape=[time_steps_q, batch_size, time_steps_k, num_features]
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)

        if attn_mask is not None:
            transposed_mask = \
                tf.transpose(tf.tile(attn_mask, [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1, 1]),
                             [2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        expanded_values = tf.expand_dims(values, axis=1)
        weighted_memories = \
            tf.reduce_sum(tf.multiply(tf.transpose(attn_weights, [1, 0, 2, 3]), expanded_values), axis=2)
        return weighted_memories

    def _multiplicative_attn(self, queries, keys, values, attn_mask):
        """ Uses multiplicative attention to compute contextually enriched source-side representations. """
        # Account for beam-search
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes time-step-wise attention scores. """
            query = tf.expand_dims(query, 1)
            return tf.multiply(keys, query)

        # Obtain attention scores
        transposed_queries = tf.transpose(queries, [1, 0, 2])  # time-major
        # attn_logits has shape=[time_steps_q, batch_size, time_steps_k, num_features]
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)

        if attn_mask is not None:
            transposed_mask = \
                tf.transpose(tf.tile(attn_mask, [get_shape_list(queries)[0] // get_shape_list(attn_mask)[0], 1, 1, 1]),
                             [2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')
        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        expanded_values = tf.expand_dims(values, axis=1)
        weighted_memories = \
            tf.reduce_sum(tf.multiply(tf.transpose(attn_weights, [1, 0, 2, 3]), expanded_values), axis=2)
        return weighted_memories

    def _attn(self, queries, keys, values, attn_mask):
        """ For each encoder layer, weighs and combines time-step-wise hidden representation into a single layer
        context state.  -- DEPRECATED, SINCE IT'S SLOW AND PROBABLY NOT ENTIRELY CORRECT """
        # Account for beam-search
        num_beams = tf.shape(queries)[0] // tf.shape(keys)[0]
        keys = tf.tile(keys, [num_beams, 1, 1])
        values = tf.tile(values, [num_beams, 1, 1])

        def _logits_fn(query):
            """ Computes position-wise attention scores. """
            query = tf.expand_dims(query, 1)
            # return tf.squeeze(self.attn_weight * (tf.nn.tanh(keys + query + norm_bias)), axis=2)
            return self.attn_weight * tf.nn.tanh(keys + query)  # 4D output

        def _weighting_fn(step_weights):
            """ Computes position-wise context vectors. """
            # step_weights = tf.expand_dims(step_weights, 2)
            return tf.reduce_sum(tf.multiply(step_weights, values), axis=1)

        # Obtain attention scores
        transposed_queries = tf.transpose(queries, [1, 0, 2])
        attn_logits = tf.map_fn(_logits_fn, transposed_queries)  # multiple queries per step are possible
        if attn_mask is not None:
            # attn_logits has shape=[batch, query_lengh, key_length, attn_features]
            transposed_mask = \
                tf.transpose(tf.tile(attn_mask, [tf.shape(queries)[0] // tf.shape(attn_mask)[0], 1, 1, 1]),
                             [2, 0, 3, 1])
            attn_logits += transposed_mask

        # Compute the attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-2, name='attn_weights')

        # Optionally apply dropout
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)

        # Obtain context vectors
        weighted_memories = tf.map_fn(_weighting_fn, attn_weights)
        weighted_memories = tf.transpose(weighted_memories, [1, 0, 2])
        return weighted_memories
