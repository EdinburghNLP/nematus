"""
Layer definitions
"""

from initializers import ortho_weight, norm_weight
import tensorflow as tf
import numpy
import sys

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
            W = tf.Variable(norm_weight(in_size, out_size), name='W')
        self.W = W
        self.b = tf.Variable(numpy.zeros((out_size,)).astype('float32'), name='b')
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
            y = self.layer_norm.forward(y, input_is_3d=input_is_3d)
        y = self.non_linearity(y)
        return y


class EmbeddingLayer(object):
    def __init__(self, vocabulary_sizes, dim_per_factor):
        assert len(vocabulary_sizes) == len(dim_per_factor)
        self.embedding_matrices = [
            tf.Variable(norm_weight(vocab_size, dim), name='embeddings')
                for vocab_size, dim in zip(vocabulary_sizes, dim_per_factor)]

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
        new_mean = numpy.zeros(shape=[layer_size], dtype=numpy.float32)
        self.new_mean = tf.Variable(new_mean,
                                    dtype=tf.float32,
                                    name='new_mean')
        new_std = numpy.ones(shape=[layer_size], dtype=numpy.float32)
        self.new_std = tf.Variable(new_std,
                                   dtype=tf.float32,
                                   name='new_std')
        self.eps = eps
    def forward(self, x, input_is_3d=False):
        # NOTE: tf.nn.moments does not support axes=[-1] or axes=[tf.rank(x)-1] :-(
        # TODO: Actually, this is probably fixed now and should be tested with latest
        # TF version. See: https://github.com/tensorflow/tensorflow/issues/8101
        axis = 2 if input_is_3d else 1
        m, v = tf.nn.moments(x, axes=[axis], keep_dims=True)
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
                 nematus_compat=False,
                 dropout_input=None,
                 dropout_state=None):
        self.state_to_gates = tf.Variable(
                                numpy.concatenate(
                                    [ortho_weight(state_size),
                                     ortho_weight(state_size)],
                                    axis=1), 
                                name='state_to_gates')
        if input_size > 0:
            self.input_to_gates = tf.Variable(
                                    numpy.concatenate(
                                        [norm_weight(input_size, state_size),
                                         norm_weight(input_size, state_size)],
                                        axis=1),
                                    name='input_to_gates')
        else:
            dropout_input = None
        self.gates_bias = tf.Variable(
                            numpy.zeros((2*state_size,)).astype('float32'),
                            name='gates_bias')

        self.state_to_proposal = tf.Variable(
                                    ortho_weight(state_size),
                                    name = 'state_to_proposal')
        if input_size > 0:
            self.input_to_proposal = tf.Variable(
                                        norm_weight(input_size, state_size),
                                        name = 'input_to_proposal')
        self.proposal_bias = tf.Variable(
                                    numpy.zeros((state_size,)).astype('float32'),
                                    name='proposal_bias')
        self.nematus_compat = nematus_compat
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            with tf.name_scope('gates_state_norm'):
                self.gates_state_norm = LayerNormLayer(2*state_size)
            with tf.name_scope('proposal_state_norm'):
                self.proposal_state_norm = LayerNormLayer(state_size)
            if input_size > 0:
                with tf.name_scope('gates_x_norm'):
                    self.gates_x_norm = LayerNormLayer(2*state_size)
                with tf.name_scope('proposal_x_norm'):
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
        if not self.nematus_compat:
            gates_x += self.gates_bias
        if self.use_layer_norm:
            gates_x = self.gates_x_norm.forward(gates_x, input_is_3d=input_is_3d)
        return gates_x

    def _get_gates_state(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_gates)
        gates_state = tf.matmul(prev_state, self.state_to_gates)
        if self.nematus_compat:
            gates_state += self.gates_bias
        if self.use_layer_norm:
            gates_state = self.gates_state_norm.forward(gates_state)
        return gates_state

    def _get_proposal_x(self,x, input_is_3d=False):
        x = apply_dropout_mask(x, self.dropout_mask_input_to_proposal,
                               input_is_3d)
        if input_is_3d: 
            proposal_x = matmul3d(x, self.input_to_proposal)
        else:
            proposal_x = tf.matmul(x, self.input_to_proposal)
        if not self.nematus_compat:
            proposal_x += self.proposal_bias
        if self.use_layer_norm:
            proposal_x = self.proposal_x_norm.forward(proposal_x, input_is_3d=input_is_3d)
        return proposal_x

    def _get_proposal_state(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_proposal)
        proposal_state = tf.matmul(prev_state, self.state_to_proposal)
        # placing the bias here is unorthodox, but we're keeping this behavior for compatibility with dl4mt-tutorial
        if self.nematus_compat:
            proposal_state += self.proposal_bias
        if self.use_layer_norm:
            proposal_state = self.proposal_state_norm.forward(proposal_state)
        return proposal_state

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

        if gates_x == None:
            gates = gates_state
        else:
            gates = gates_x + gates_state
        gates = tf.nn.sigmoid(gates)
        read_gate, update_gate = tf.split(gates,
                                          num_or_size_splits=2,
                                          axis=1)

        proposal = proposal_state*read_gate
        if proposal_x != None:
            proposal += proposal_x
        proposal = tf.tanh(proposal)
        new_state = update_gate*prev_state + (1-update_gate)*proposal

        return new_state

class DeepTransitionGRUStep(object):
    def __init__(self,
                 input_size,
                 state_size,
                 batch_size,
                 use_layer_norm=False,
                 nematus_compat=False,
                 dropout_input=None,
                 dropout_state=None,
                 transition_depth=1,
                 name_scope_fn=lambda i: "gru{0}".format(i)):
        self.gru_steps = []
        for i in range(transition_depth):
            with tf.name_scope(name_scope_fn(i)):
                gru = GRUStep(input_size=(input_size if i == 0 else 0),
                              state_size=state_size,
                              batch_size=batch_size,
                              use_layer_norm=use_layer_norm,
                              nematus_compat=nematus_compat,
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
                 nematus_compat=False,
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
            with tf.name_scope("level{0}".format(i)):
                self.grus.append(DeepTransitionGRUStep(
                    input_size=in_size,
                    state_size=state_size,
                    batch_size=batch_size,
                    use_layer_norm=use_layer_norm,
                    nematus_compat=nematus_compat,
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
    def forward(self, x, x_mask=None, context_layer=None):

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
            if i == 0:
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
                 dropout_state=None,
                 projection_dim=-1,
                 n_attention_heads=1):
        self.state_to_hidden = tf.Variable(
                                norm_weight(state_size, hidden_size),
                                name='state_to_hidden')
        self.context_to_hidden = tf.Variable( #TODO: Nematus uses ortho_weight here - important?
                                    norm_weight(context_state_size, hidden_size), 
                                    name='context_to_hidden')
        self.hidden_bias = tf.Variable(
                            numpy.zeros((hidden_size,)).astype('float32'),
                            name='hidden_bias')
        self.hidden_to_score = tf.Variable(
                                norm_weight(hidden_size, n_attention_heads),
                                name='hidden_to_score')
        self.n_attention_heads=n_attention_heads
        self.use_layer_norm = use_layer_norm
        self.apply_value_projection = (projection_dim != -1)
        if self.apply_value_projection:
            self.value_projection = tf.Variable(
                                    norm_weight(context_state_size, projection_dim*n_attention_heads),
                                    name='value_projection')
            self.attention_context_size = projection_dim * n_attention_heads
        else:
            self.attention_context_size = context_state_size * n_attention_heads
        

        if self.use_layer_norm:
            with tf.name_scope('hidden_context_norm'):
                self.hidden_context_norm = LayerNormLayer(layer_size=hidden_size)
            with tf.name_scope('hidden_state_norm'):
                self.hidden_state_norm = LayerNormLayer(layer_size=hidden_size)
            if self.apply_value_projection:
                with tf.name_scope('value_projection_norm'):
                    self.value_projection_norm = LayerNormLayer(layer_size=projection_dim*n_attention_heads)
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
                self.hidden_context_norm.forward(self.hidden_from_context, input_is_3d=True)

        
        if self.apply_value_projection:
            self.context_value = matmul3d(self.context, self.value_projection)
            if self.use_layer_norm:
                self.context_value = self.value_projection_norm.forward(self.context_value, input_is_3d=True)
            self.context_value = tf.reshape(self.context_value, [tf.shape(self.context_value)[0], -1, n_attention_heads, projection_dim])
        elif n_attention_heads == 1:
            self.context_value = tf.expand_dims(self.context, axis=2)
        else:
            self.context_value = tf.tile(tf.expand_dims(self.context, axis=2), tf.constant([1, 1, n_attention_heads, 1]))

    def forward(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_hidden)
        hidden_from_state = tf.matmul(prev_state, self.state_to_hidden)
        if self.use_layer_norm:
            hidden_from_state = \
                self.hidden_state_norm.forward(hidden_from_state, input_is_3d=False)
        hidden = self.hidden_from_context + hidden_from_state
        hidden = tf.nn.tanh(hidden)
        # context has shape seqLen x batch x context_state_size
        # mask has shape seqLen x batch

        scores = matmul3d(hidden, self.hidden_to_score) # seqLen x batch x heads
        #scores = tf.squeeze(scores, axis=2)
        scores = scores - tf.reduce_max(scores, axis=0, keepdims=True)
        scores = tf.exp(scores)
        scores *= tf.expand_dims(self.context_mask, axis=2)
        scores = scores / tf.reduce_sum(scores, axis=0, keepdims=True)

        #attention_context = self.context_value * tf.expand_dims(scores, axis=2)
        attention_context = self.context_value * tf.expand_dims(scores, axis=3)
        attention_context = tf.reduce_sum(attention_context, axis=0, keepdims=False)
        attention_context = tf.reshape(attention_context, [-1, self.attention_context_size])

        return attention_context

class DotProductAttentionStep(object):
    def __init__(self,
                 context,
                 context_state_size,
                 context_mask,
                 state_size,
                 hidden_size, 		# key projection size
                 use_layer_norm=False,
                 dropout_context=None,
                 dropout_state=None,
                 projection_dim=-1,     # value projection size
                 n_attention_heads=1):
        self.state_to_hidden = tf.Variable(
                                norm_weight(state_size, hidden_size*n_attention_heads),
                                name='state_to_hidden')
        self.context_to_hidden = tf.Variable( 
                                    norm_weight(context_state_size, hidden_size*n_attention_heads), 
                                    name='context_to_hidden')
        self.n_attention_heads=n_attention_heads
        self.use_layer_norm = use_layer_norm
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim if projection_dim != -1 else self.hidden_size
        self.value_projection = tf.Variable(
                                 norm_weight(context_state_size, projection_dim*n_attention_heads),
                                 name='value_projection')
        self.attention_context_size = projection_dim * n_attention_heads

        if self.use_layer_norm:
            with tf.name_scope('value_projection_norm'):
                self.value_projection_norm = LayerNormLayer(layer_size=projection_dim*n_attention_heads)

        self.context = context
        self.context_mask = tf.transpose(context_mask) 								# batch_size x seqLen

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
        self.hidden_from_context = tf.reshape(self.hidden_from_context, [tf.shape(self.hidden_from_context)[0], -1, n_attention_heads, hidden_size])
        self.hidden_from_context = tf.transpose(self.hidden_from_context, [1, 2, 3, 0]) 	  		# batch_size x n_attention_heads x hidden_size x seqLen
        
        self.context_value = matmul3d(self.context, self.value_projection)
        if self.use_layer_norm:
            self.context_value = self.value_projection_norm.forward(self.context_value, input_is_3d=True)
        self.context_value = tf.reshape(self.context_value, [tf.shape(self.context_value)[0], -1, n_attention_heads, projection_dim])

    def forward(self, prev_state):
        prev_state = apply_dropout_mask(prev_state,
                                        self.dropout_mask_state_to_hidden)
        hidden_from_state = tf.matmul(prev_state, self.state_to_hidden)
        hidden_from_state = tf.reshape(hidden_from_state, [-1, self.n_attention_heads, 1, self.hidden_size]) 	# batch_size x n_attention_heads x 1 x hidden_size

        scores = tf.matmul(hidden_from_state, self.hidden_from_context)
        scores = tf.squeeze(scores, axis=2)									# batch_size x n_attention_heads x seqLen

        scores = scores - tf.reduce_max(scores, axis=2, keepdims=True)
        scores = tf.exp(scores)
        scores *= tf.expand_dims(self.context_mask, axis=1)
        scores = scores / tf.reduce_sum(scores, axis=2, keepdims=True)

        attention_context = self.context_value * tf.expand_dims(tf.transpose(scores, [2, 0, 1]), axis=3)
        attention_context = tf.reduce_sum(attention_context, axis=0, keepdims=False)
        attention_context = tf.reshape(attention_context, [-1, self.attention_context_size])

        return attention_context


class DeepTransitionRNNWithMultiHopAttentionStep(object):
    def __init__(self,
                 context,
                 context_state_size,
                 context_mask,
                 state_size,
                 batch_size,
                 rnn_transition_depth=1,
                 n_attention_hops = 1,
                 attention_step_options={},
                 rnn_step_options={},
                 attention_step_class = AttentionStep,
                 rnn_step_class = GRUStep,
                 attention_name_scope_fn=lambda i: ("attention{0}".format(i) if (i > 0) else "attention"),                 
                 rnn_name_scope_fn=lambda i: "gru{0}".format(i)):
        self.context = context
        self.context_state_size = context_state_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.rnn_transition_depth = rnn_transition_depth
        self.n_attention_hops = n_attention_hops
        self.attention_step_options = attention_step_options
        self.rnn_step_options = rnn_step_options
        self.attention_step_class = attention_step_class
        self.rnn_step_class = rnn_step_class

        self.attention_steps = []
        self.rnn_steps = []
        for i in range(rnn_transition_depth):
            if (i < n_attention_hops):
                with tf.name_scope(attention_name_scope_fn(i)):
                    attention_step = attention_step_class(context=context,
                                                          context_state_size=context_state_size,
                                                          context_mask=context_mask,
                                                          state_size=state_size,
                                                          **attention_step_options)
                    self.attention_steps.append(attention_step)
            with tf.name_scope(rnn_name_scope_fn(i)):
                rnn = rnn_step_class(input_size=(attention_step.attention_context_size if i < n_attention_hops else 0),
                                     state_size=state_size,
                                     batch_size=batch_size,
                                     **rnn_step_options)
            self.rnn_steps.append(rnn)
        if n_attention_hops > 0:
            self.attention_context_size = self.attention_steps[-1].attention_context_size
         

    def precompute_from_x(self, x):
        raise NotImplementedError()
        #return self.gru_steps[0].precompute_from_x(x)

    def forward(self,
                prev_state):
        new_state = prev_state
        attention_context = None
        for i, rnn_step in enumerate(self.rnn_steps):
            if i < self.n_attention_hops:
                attention_context = self.attention_steps[i].forward(prev_state=new_state)
                x = attention_context
            else:
                x = None
            new_state = rnn_step.forward(x=x, prev_state=new_state)
        return attention_context, new_state
        

class Masked_cross_entropy_loss(object):
    def __init__(self,
                 y_true,
                 y_mask):
        self.y_true = y_true
        self.y_mask = y_mask


    def forward(self, logits):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_true,
                logits=logits)
        #cost has shape seqLen x batch
        cost *= self.y_mask
        cost = tf.reduce_sum(cost, axis=0, keepdims=False)
        return cost

class PReLU(object):
    def __init__(self,
                 in_size,
                 initial_slope = 1.0):

        self.slope = tf.Variable(initial_slope * numpy.ones((in_size,)).astype('float32'), name='slope')

    def forward(self, x):
        pos = tf.nn.relu(x)
        neg = x - pos
        y = pos + self.slope * neg
        return y
