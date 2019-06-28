"""
Layer definitions
"""

import tensorflow as tf

import initializers

"""Controls bias and layer normalization handling in GRUs."""
class LegacyBiasType:
    """
    THEANO_A and THEANO_B are for backwards compatibility with models that
    were trained with the Theano version of Nematus.

    THEANO_A matches the behaviour of Theano's gru_layer (used in the encoder
    and the high levels of the decoder) and also of the pre-attention sub-layer
    of gru_cond_layer (used in the base level of the decoder).

    THEANO_B matches the behaviour of the post-attention sub-layer of Theano's
    gru_cond_layer (used in the base level of the decoder).

    NEMATUS_COMPAT_TRUE and NEMATUS_COMPAT_FALSE should be used for all models
    trained with the TensorFlow version of Nematus. For shallow models, bias
    and layer normalization handling is identical to the Theano version. For
    deep models, the behaviour was (inadvertently) changed. Empirically, it
    doesn't seem to make much difference, as long as one type is used
    consistently at training and test times.
    """
    THEANO_A = 1
    THEANO_B = 2
    NEMATUS_COMPAT_TRUE = 3
    NEMATUS_COMPAT_FALSE = 4


def matmul3d(x3d, matrix):
    shape = tf.shape(x3d)
    mat_shape = tf.shape(matrix)
    x2d = tf.reshape(x3d, [shape[0]*shape[1], shape[2]])
    result2d = tf.matmul(x2d, matrix)
    result3d = tf.reshape(result2d, [shape[0], shape[1], mat_shape[1]])
    return result3d

def apply_dropout_mask(x, mask, input_is_3d=False):
    if mask == None:
        return x
    if input_is_3d:
        mask_3d = tf.expand_dims(mask, 0)
        mask_3d = tf.tile(mask_3d, [tf.shape(x)[0], 1, 1])
        return tf.multiply(x, mask_3d)
    else:
        return tf.multiply(x, mask)

class FeedForwardLayer(object):
    def __init__(self,
                 in_size,
                 out_size,
                 batch_size,
                 non_linearity=tf.nn.tanh,
                 W=None,
                 use_layer_norm=False,
                 dropout_input=None):
        if W is None:
            init = initializers.norm_weight(in_size, out_size)
            W = tf.get_variable('W', initializer=init)
        self.W = W
        self.b = tf.get_variable('b', [out_size],
                                 initializer=tf.zeros_initializer)
        self.non_linearity = non_linearity
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNormLayer(layer_size=out_size)
        # Create a dropout mask for input values (reused at every timestep).
        if dropout_input == None:
            self.dropout_mask = None
        else:
            ones = tf.ones([batch_size, in_size])
            self.dropout_mask = dropout_input(ones)


    def forward(self, x, input_is_3d=False):
        x = apply_dropout_mask(x, self.dropout_mask, input_is_3d)
        if input_is_3d:
            y = matmul3d(x, self.W) + self.b
        else:
            y = tf.matmul(x, self.W) + self.b
        if self.use_layer_norm:
            y = self.layer_norm.forward(y)
        y = self.non_linearity(y)
        return y


class EmbeddingLayer(object):
    def __init__(self, vocabulary_sizes, dim_per_factor):
        assert len(vocabulary_sizes) == len(dim_per_factor)
        self.embedding_matrices = []
        for i in range(len(vocabulary_sizes)):
            vocab_size, dim = vocabulary_sizes[i], dim_per_factor[i]
            var_name = 'embeddings' if i == 0 else 'embeddings_' + str(i)
            init = initializers.norm_weight(vocab_size, dim)
            matrix = tf.get_variable(var_name, initializer=init)
            self.embedding_matrices.append(matrix)

    def forward(self, x, factor=None):
        if factor == None:
            # Assumes that x has shape: factors, ...
            embs = [tf.nn.embedding_lookup(matrix, x[i])
                    for i, matrix in enumerate(self.embedding_matrices)]
            return tf.concat(embs, axis=-1)
        else:
            matrix = self.embedding_matrices[factor]
            return tf.nn.embedding_lookup(matrix, x)

    def get_embeddings(self, factor=None):
        if factor == None:
            return self.embedding_matrices
        else:
            return self.embedding_matrices[factor]


class RecurrentLayer(object):
    def __init__(self,
                 initial_state,
                 step_fn):
        self.initial_state = initial_state
        self.step_fn = step_fn

    def forward(self, x):
        # Assumes that x has shape: time, batch, ...
        states = tf.scan(fn=self.step_fn,
                         elems=x,
                         initializer=self.initial_state)
        return states

class LayerNormLayer(object):
    def __init__(self,
                 layer_size,
                 eps=1e-5):
        #TODO: If nematus_compat is true, then eps must be 1e-5!
        self.new_mean = tf.get_variable('new_mean', [layer_size],
                                        initializer=tf.zeros_initializer)
        self.new_std = tf.get_variable('new_std', [layer_size],
                                       initializer=tf.constant_initializer(1))
        self.eps = eps

    def forward(self, x):
        m, v = tf.nn.moments(x, axes=[-1], keep_dims=True)
        std = tf.sqrt(v + self.eps)
        norm_x = (x-m)/std
        new_x = norm_x*self.new_std + self.new_mean
        return new_x

class GRUStep(object):
    def __init__(self, 
                 input_size, 
                 state_size,
                 batch_size,
                 use_layer_norm=False,
                 legacy_bias_type=LegacyBiasType.NEMATUS_COMPAT_FALSE,
                 dropout_input=None,
                 dropout_state=None):
        init = tf.concat([initializers.ortho_weight(state_size),
                          initializers.ortho_weight(state_size)],
                         axis=1)
        self.state_to_gates = tf.get_variable('state_to_gates',
                                              initializer=init)
        if input_size > 0:
            init = tf.concat([initializers.norm_weight(input_size, state_size),
                              initializers.norm_weight(input_size, state_size)],
                             axis=1)
            self.input_to_gates = tf.get_variable('input_to_gates',
                                                  initializer=init)

        if input_size == 0 and legacy_bias_type == LegacyBiasType.NEMATUS_COMPAT_FALSE:
            self.gates_bias = None
        else:
            self.gates_bias = tf.get_variable('gates_bias', [2*state_size],
                                          initializer=tf.zeros_initializer)

        init = initializers.ortho_weight(state_size)
        self.state_to_proposal = tf.get_variable('state_to_proposal',
                                                 initializer=init)
        if input_size > 0:
            init = initializers.norm_weight(input_size, state_size)
            self.input_to_proposal = tf.get_variable('input_to_proposal',
                                                     initializer=init)

        if input_size == 0 and legacy_bias_type == LegacyBiasType.NEMATUS_COMPAT_FALSE:
            self.proposal_bias = None
        else:
            self.proposal_bias = tf.get_variable('proposal_bias', [state_size],
                                             initializer=tf.zeros_initializer)

        self.legacy_bias_type = legacy_bias_type
        self.use_layer_norm = use_layer_norm

        self.gates_state_norm = None
        self.proposal_state_norm = None
        self.gates_x_norm = None
        self.proposal_x_norm = None
        if self.use_layer_norm:
            with tf.variable_scope('gates_state_norm'):
                self.gates_state_norm = LayerNormLayer(2*state_size)
            with tf.variable_scope('proposal_state_norm'):
                self.proposal_state_norm = LayerNormLayer(state_size)
            if input_size > 0:
                with tf.variable_scope('gates_x_norm'):
                    self.gates_x_norm = LayerNormLayer(2*state_size)
                with tf.variable_scope('proposal_x_norm'):
                    self.proposal_x_norm = LayerNormLayer(state_size)

        # Create dropout masks for input values (reused at every timestep).
        if dropout_input == None:
            self.dropout_mask_input_to_gates = None
            self.dropout_mask_input_to_proposal = None
        else:
            ones = tf.ones([batch_size, input_size])
            self.dropout_mask_input_to_gates = dropout_input(ones)
            self.dropout_mask_input_to_proposal = dropout_input(ones)

        # Create dropout masks for state values (reused at every timestep).
        if dropout_state == None:
            self.dropout_mask_state_to_gates = None
            self.dropout_mask_state_to_proposal = None
        else:
            ones = tf.ones([batch_size, state_size])
            self.dropout_mask_state_to_gates = dropout_state(ones)
            self.dropout_mask_state_to_proposal = dropout_state(ones)

    def _get_gates_x(self, x, input_is_3d=False):
        x = apply_dropout_mask(x, self.dropout_mask_input_to_gates, input_is_3d)
        if input_is_3d:
            gates_x = matmul3d(x, self.input_to_gates)
        else:
            gates_x = tf.matmul(x, self.input_to_gates)
        return self._layer_norm_and_bias(x=gates_x,
                                         b=self.gates_bias,
                                         layer_norm=self.gates_x_norm,
                                         x_is_input=True)

    def _get_gates_state(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_gates)
        gates_state = tf.matmul(prev_state, self.state_to_gates)
        return self._layer_norm_and_bias(x=gates_state,
                                         b=self.gates_bias,
                                         layer_norm=self.gates_state_norm,
                                         x_is_input=False)

    def _get_proposal_x(self,x, input_is_3d=False):
        x = apply_dropout_mask(x, self.dropout_mask_input_to_proposal,
                               input_is_3d)
        if input_is_3d: 
            proposal_x = matmul3d(x, self.input_to_proposal)
        else:
            proposal_x = tf.matmul(x, self.input_to_proposal)
        return self._layer_norm_and_bias(x=proposal_x,
                                         b=self.proposal_bias,
                                         layer_norm=self.proposal_x_norm,
                                         x_is_input=True)

    def _get_proposal_state(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_proposal)
        proposal_state = tf.matmul(prev_state, self.state_to_proposal)
        return self._layer_norm_and_bias(x=proposal_state,
                                         b=self.proposal_bias,
                                         layer_norm=self.proposal_state_norm,
                                         x_is_input=False)

    def _layer_norm_and_bias(self, x, b, layer_norm, x_is_input):
        assert self.use_layer_norm == (layer_norm is not None)
        if (self.legacy_bias_type == LegacyBiasType.THEANO_A
            or self.legacy_bias_type == LegacyBiasType.NEMATUS_COMPAT_FALSE):
            if x_is_input:
                return layer_norm.forward(x+b) if self.use_layer_norm else x+b
            else:
                return layer_norm.forward(x) if self.use_layer_norm else x
        elif (self.legacy_bias_type == LegacyBiasType.THEANO_B
              or self.legacy_bias_type == LegacyBiasType.NEMATUS_COMPAT_TRUE):
            if x_is_input:
                return layer_norm.forward(x) if self.use_layer_norm else x
            else:
                return layer_norm.forward(x+b) if self.use_layer_norm else x+b
        else:
            assert False

    def precompute_from_x(self, x):
        # compute gates_x and proposal_x in one big matrix multiply
        # if x is fully known upfront 
        # this method exists only for efficiency reasons

        gates_x = self._get_gates_x(x, input_is_3d=True)
        proposal_x = self._get_proposal_x(x, input_is_3d=True)
        return gates_x, proposal_x

    def forward(self,
                prev_state,
                x=None,
                gates_x=None,
                gates_state=None,
                proposal_x=None,
                proposal_state=None):
        if gates_x is None and x != None:
            gates_x = self._get_gates_x(x) 
        if proposal_x is None and x != None:
            proposal_x = self._get_proposal_x(x) 
        if gates_state is None:
            gates_state = self._get_gates_state(prev_state) 
        if proposal_state is None:
            proposal_state = self._get_proposal_state(prev_state) 

        if gates_x != None:
            # level l = 0 in deep transition GRU
            gates = gates_x + gates_state
        else:
            # level l > 0 in deep transition GRU
            gates = gates_state
            if self.legacy_bias_type == LegacyBiasType.THEANO_A:
                gates += self.gates_bias
        gates = tf.nn.sigmoid(gates)
        read_gate, update_gate = tf.split(gates,
                                          num_or_size_splits=2,
                                          axis=1)

        proposal = proposal_state*read_gate
        if proposal_x != None:
            # level l = 0 in deep transition GRU
            proposal += proposal_x
        else:
            # level l > 0 in deep transition GRU
            if self.legacy_bias_type == LegacyBiasType.THEANO_A:
                proposal += self.proposal_bias
        proposal = tf.tanh(proposal)
        new_state = update_gate*prev_state + (1-update_gate)*proposal

        return new_state

class DeepTransitionGRUStep(object):
    def __init__(self,
                 input_size,
                 state_size,
                 batch_size,
                 use_layer_norm=False,
                 legacy_bias_type=LegacyBiasType.NEMATUS_COMPAT_FALSE,
                 dropout_input=None,
                 dropout_state=None,
                 transition_depth=1,
                 var_scope_fn=lambda i: "gru{0}".format(i)):
        self.gru_steps = []
        for i in range(transition_depth):
            with tf.variable_scope(var_scope_fn(i)):
                gru = GRUStep(input_size=(input_size if i == 0 else 0),
                              state_size=state_size,
                              batch_size=batch_size,
                              use_layer_norm=use_layer_norm,
                              legacy_bias_type=legacy_bias_type,
                              dropout_input=(dropout_input if i == 0 else None),
                              dropout_state=dropout_state)
            self.gru_steps.append(gru)

    def precompute_from_x(self, x):
        return self.gru_steps[0].precompute_from_x(x)

    def forward(self,
                prev_state,
                x=None,
                gates_x=None,
                gates_state=None,
                proposal_x=None,
                proposal_state=None):
        new_state = self.gru_steps[0].forward(prev_state=prev_state,
                                              x=x,
                                              gates_x=gates_x,
                                              gates_state=gates_state,
                                              proposal_x=proposal_x,
                                              proposal_state=proposal_state)
        for gru_step in self.gru_steps[1:]:
            new_state = gru_step.forward(prev_state=new_state)
        return new_state


class GRUStack(object):
    def __init__(self,
                 input_size,
                 state_size,
                 batch_size,
                 use_layer_norm=False,
                 legacy_bias_type=LegacyBiasType.NEMATUS_COMPAT_FALSE,
                 dropout_input=None,
                 dropout_state=None,
                 stack_depth=1,
                 transition_depth=1,
                 alternating=False,
                 reverse_alternation=False,
                 context_state_size=0,
                 residual_connections=False,
                 first_residual_output=0):
        self.state_size = state_size
        self.batch_size = batch_size
        self.alternating = alternating
        self.reverse_alternation = reverse_alternation
        self.context_state_size = context_state_size
        self.residual_connections = residual_connections
        self.first_residual_output = first_residual_output
        self.grus = []
        for i in range(stack_depth):
            in_size = (input_size if i == 0 else state_size) + context_state_size
            with tf.variable_scope("level{0}".format(i)):
                self.grus.append(DeepTransitionGRUStep(
                    input_size=in_size,
                    state_size=state_size,
                    batch_size=batch_size,
                    use_layer_norm=use_layer_norm,
                    legacy_bias_type=legacy_bias_type,
                    dropout_input=(dropout_input if i == 0 else dropout_state),
                    dropout_state=dropout_state,
                    transition_depth=transition_depth))

    # Single timestep version
    def forward_single(self, prev_states, x, context=None):
        stack_depth = len(self.grus)
        states = [None] * stack_depth
        for i in range(stack_depth):
            if context == None:
                x2 = x
            else:
                x2 = tf.concat([x, context], axis=-1)
            states[i] = self.grus[i].forward(prev_states[i], x2)
            if not self.residual_connections or i < self.first_residual_output:
                x = states[i]
            else:
                x += states[i]
        return x, states

    # Layer version
    def forward(self, x, x_mask=None, context_layer=None, init_state=None):

        assert not (self.reverse_alternation and x_mask == None)

#        assert (context_layer == None or
#                tf.shape(context_layer)[-1] == self.context_state_size)

        def create_step_fun(gru):
            def step_fn(prev_state, x):
                gates_x2d, proposal_x2d = x[0], x[1]
                new_state = gru.forward(prev_state,
                                        gates_x=gates_x2d,
                                        proposal_x=proposal_x2d)
                if len(x) > 2:
                    mask = x[2]
                    new_state *= mask # batch x 1
                    # first couple of states of reversed encoder should be zero
                    # this is why we need to multiply by mask
                    # this way, when the reversed encoder reaches actual words
                    # the state will be zeros and not some accumulated garbage
                return new_state
            return step_fn

        if init_state is None:
            init_state = tf.zeros(shape=[self.batch_size, self.state_size],
                              dtype=tf.float32)
        if x_mask != None:
            x_mask_r = tf.reverse(x_mask, axis=[0])
            x_mask_bwd = tf.expand_dims(x_mask_r, axis=[2]) #seqLen x batch x 1

        for i, gru in enumerate(self.grus):
            layer = RecurrentLayer(initial_state=init_state,
                                   step_fn=create_step_fun(gru))
            if context_layer == None:
                x2 = x
            else:
                x2 = tf.concat([x, context_layer], axis=-1)
            if not self.alternating:
                left_to_right = True
            else:
                if self.reverse_alternation:
                    left_to_right = (i % 2 == 1)
                else:
                    left_to_right = (i % 2 == 0)
            if left_to_right:
                # Recurrent state flows from left to right in this layer.
                gates_x, proposal_x = gru.precompute_from_x(x2)
                h = layer.forward((gates_x, proposal_x))
            else:
                # Recurrent state flows from right to left in this layer.
                x2_reversed = tf.reverse(x2, axis=[0])
                gates_x, proposal_x = gru.precompute_from_x(x2_reversed)
                h_reversed = layer.forward((gates_x, proposal_x, x_mask_bwd))
                h = tf.reverse(h_reversed, axis=[0])
            # Compute the word states, which will become the input for the
            # next layer (or the output of the stack if we're at the top).
            if not self.residual_connections or i < self.first_residual_output:
                x = h
            else:
                x += h # Residual connection
        return x


class AttentionStep(object):
    def __init__(self,
                 context,
                 context_state_size,
                 context_mask,
                 state_size,
                 hidden_size,
                 use_layer_norm=False,
                 dropout_context=None,
                 dropout_state=None):
        init = initializers.norm_weight(state_size, hidden_size)
        self.state_to_hidden = tf.get_variable('state_to_hidden',
                                               initializer=init)
        #TODO: Nematus uses ortho_weight here - important?
        init = initializers.norm_weight(context_state_size, hidden_size)
        self.context_to_hidden = tf.get_variable('context_to_hidden',
                                                 initializer=init)
        self.hidden_bias = tf.get_variable('hidden_bias', [hidden_size],
                                           initializer=tf.zeros_initializer)
        init = initializers.norm_weight(hidden_size, 1)
        self.hidden_to_score = tf.get_variable('hidden_to_score',
                                               initializer=init)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            with tf.variable_scope('hidden_context_norm'):
                self.hidden_context_norm = LayerNormLayer(layer_size=hidden_size)
            with tf.variable_scope('hidden_state_norm'):
                self.hidden_state_norm = LayerNormLayer(layer_size=hidden_size)
        self.context = context
        self.context_mask = context_mask

        batch_size = tf.shape(context)[1]

        # Create a dropout mask for context values (reused at every timestep).
        if dropout_context == None:
            self.dropout_mask_context_to_hidden = None
        else:
            ones = tf.ones([batch_size, context_state_size])
            self.dropout_mask_context_to_hidden = dropout_context(ones)

        # Create a dropout mask for state values (reused at every timestep).
        if dropout_state == None:
            self.dropout_mask_state_to_hidden = None
        else:
            ones = tf.ones([batch_size, state_size])
            self.dropout_mask_state_to_hidden = dropout_state(ones)

        # precompute these activations, they are the same at each step
        # Ideally the compiler would have figured out that too
        context = apply_dropout_mask(context,
                                     self.dropout_mask_context_to_hidden, True)
        self.hidden_from_context = matmul3d(context, self.context_to_hidden)
        self.hidden_from_context += self.hidden_bias
        if self.use_layer_norm:
            self.hidden_from_context = \
                self.hidden_context_norm.forward(self.hidden_from_context)

    def forward(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_hidden)
        hidden_from_state = tf.matmul(prev_state, self.state_to_hidden)
        if self.use_layer_norm:
            hidden_from_state = \
                self.hidden_state_norm.forward(hidden_from_state)
        hidden = self.hidden_from_context + hidden_from_state
        hidden = tf.nn.tanh(hidden)
        # context has shape seqLen x batch x context_state_size
        # mask has shape seqLen x batch

        scores = matmul3d(hidden, self.hidden_to_score) # seqLen x batch x 1
        scores = tf.squeeze(scores, axis=2)
        scores = scores - tf.reduce_max(scores, axis=0, keepdims=True)
        scores = tf.exp(scores)
        scores *= self.context_mask
        scores = scores / tf.reduce_sum(scores, axis=0, keepdims=True)

        attention_context = self.context * tf.expand_dims(scores, axis=2)
        attention_context = tf.reduce_sum(attention_context, axis=0, keepdims=False)

        return attention_context, scores

class Masked_cross_entropy_loss(object):
    def __init__(self,
                 y_true,
                 y_mask,
                 label_smoothing=0.1,
                 training=False):
        self.y_true = y_true
        self.y_mask = y_mask

        if label_smoothing:
           self.label_smoothing = True
           self.smoothing_factor = label_smoothing
        else:
           self.label_smoothing = False


    def forward(self, logits):
        if self.label_smoothing:
            uniform_prob = self.smoothing_factor / tf.cast(tf.shape(logits)[-1], tf.float32)
            smoothed_prob = 1.0-self.smoothing_factor + uniform_prob
            onehot_labels = tf.one_hot(self.y_true, tf.shape(logits)[-1], on_value = smoothed_prob, off_value = uniform_prob, dtype = tf.float32)
            cost = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=logits,
                weights=self.y_mask,
                reduction=tf.losses.Reduction.NONE)
        else:
            cost = tf.losses.sparse_softmax_cross_entropy(
                labels=self.y_true,
                logits=logits,
                weights=self.y_mask,
                reduction=tf.losses.Reduction.NONE)

        cost = tf.reduce_sum(cost, axis=0, keepdims=False)
        return cost

class LexicalModel(object):
    def __init__(self,
                 in_size,
                 out_size,
                 batch_size,
                 use_layer_norm=False,
                 dropout_embedding=None,
                 dropout_hidden=None):

        self.ff = FeedForwardLayer(
                    in_size=in_size,
                    out_size=out_size,
                    batch_size=batch_size,
                    use_layer_norm=use_layer_norm,
                    dropout_input=dropout_hidden)

        if dropout_embedding is None:
            self.dropout_mask_embedding = None
        else:
            ones = tf.ones([batch_size, in_size])
            self.dropout_mask_embedding = dropout_embedding(ones)

    def forward(self, x_embs, att_alphas, multi_step=False):
        x_embs = apply_dropout_mask(x_embs, self.dropout_mask_embedding, input_is_3d=True)
        x_emb_weighted = x_embs * tf.expand_dims(att_alphas, axis=(3 if multi_step else 2))
        x_emb_weighted = tf.nn.tanh(tf.reduce_sum(x_emb_weighted, axis=(1 if multi_step else 0), keepdims=False))
        lexical_state = self.ff.forward(x_emb_weighted, input_is_3d=multi_step) + x_emb_weighted

        return lexical_state

class PReLU(object):
    def __init__(self,
                 in_size,
                 initial_slope = 1.0):
        init = initial_slope * tf.ones([in_size])
        self.slope = tf.get_variable('slope', initializer=init)

    def forward(self, x):
        pos = tf.nn.relu(x)
        neg = x - pos
        y = pos + self.slope * neg
        return y
