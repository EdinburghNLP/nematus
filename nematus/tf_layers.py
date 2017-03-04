"""
Layer definitions
"""

from tf_initializers import ortho_weight, norm_weight
import tensorflow as tf
import numpy 

def matmul3d(x3d, matrix):
    shape = tf.shape(x3d)
    mat_shape = tf.shape(matrix)
    x2d = tf.reshape(x3d, [shape[0]*shape[1], shape[2]])
    result2d = tf.matmul(x2d, matrix)
    result3d = tf.reshape(result2d, [shape[0], shape[1], mat_shape[1]])
    return result3d

class FeedForwardLayer(object):
    def __init__(self,
                 in_size,
                 out_size,
                 non_linearity=tf.nn.tanh):
        self.W = tf.Variable(norm_weight(in_size, out_size), name='W')
        self.b = tf.Variable(numpy.zeros((out_size,)).astype('float32'), name='b')
        self.non_linearity = non_linearity

    def forward(self, x, input_is_3d=False):
        if input_is_3d:
            y = matmul3d(x, self.W) + self.b
        else:
            y = tf.matmul(x, self.W) + self.b
        y = self.non_linearity(y)
        return y

class EmbeddingLayer(object):
    def __init__(self,
                 vocabulary_size,
                 embedding_size):
        self.embeddings = tf.Variable(norm_weight(vocabulary_size, embedding_size),
                                      name='embeddings')
    
    def forward(self, x):
        embs = tf.nn.embedding_lookup(self.embeddings, x)
        return embs

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
    
class GRUStep(object):
    def __init__(self, 
                 input_size, 
                 state_size,
                 nematus_compat=False):
        self.state_to_gates = tf.Variable(
                                numpy.concatenate(
                                    [ortho_weight(state_size),
                                     ortho_weight(state_size)],
                                    axis=1), 
                                name='state_to_gates')
        self.input_to_gates = tf.Variable(
                                numpy.concatenate(
                                    [norm_weight(input_size, state_size),
                                     norm_weight(input_size, state_size)],
                                    axis=1), 
                                name='input_to_gates')
        self.gates_bias = tf.Variable(
                            numpy.zeros((2*state_size,)).astype('float32'),
                            name='gates_bias')

        self.state_to_proposal = tf.Variable(
                                    ortho_weight(state_size),
                                    name = 'state_to_proposal')
        self.input_to_proposal = tf.Variable(
                                    norm_weight(input_size, state_size),
                                    name = 'input_to_proposal')
        self.proposal_bias = tf.Variable(
                                    numpy.zeros((state_size,)).astype('float32'),
                                    name='proposal_bias')
        self.nematus_compat = nematus_compat

    def _forward(self, prev_state, gates_x, proposal_x):
        gates = tf.matmul(prev_state, self.state_to_gates)
        gates += gates_x
        gates = tf.nn.sigmoid(gates)
        read_gate, update_gate = tf.split(gates,
                                          num_or_size_splits=2,
                                          axis=1)

        proposal = tf.matmul(prev_state, self.state_to_proposal)
        if self.nematus_compat:
            proposal += self.proposal_bias
        proposal *= read_gate
        proposal += proposal_x
        proposal  = tf.tanh(proposal)

        new_state = update_gate*prev_state + (1-update_gate)*proposal

        return new_state

    def forward(self, prev_state, x):
        gates_x = tf.matmul(x, self.input_to_gates) + self.gates_bias
        if self.nematus_compat:
            proposal_x = tf.matmul(x, self.input_to_proposal)
        else:
            proposal_x = tf.matmul(x, self.input_to_proposal) + self.proposal_bias

        return self._forward(prev_state, gates_x, proposal_x)

    def _get_gates_x_proposal_x(self, x):
        gates_x = matmul3d(x, self.input_to_gates) + self.gates_bias
        proposal_x = matmul3d(x, self.input_to_proposal) + self.proposal_bias
        return gates_x, proposal_x


class AttentionStep(object):
    def __init__(self,
                 context,
                 context_state_size,
                 context_mask,
                 state_size,
                 hidden_size):
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
                                norm_weight(hidden_size, 1),
                                name='hidden_to_score')
        
        self.context = context
        self.context_mask = context_mask

        # precompute these activations, they are the same at each step
        # Ideally the compiler would have figured out that too
        self.hidden_from_context = matmul3d(context, self.context_to_hidden)
        self.hidden_from_context += self.hidden_bias

    def forward(self, prev_state):
        hidden = self.hidden_from_context
        hidden += tf.matmul(prev_state, self.state_to_hidden)
        hidden = tf.nn.tanh(hidden)
        # context has shape seqLen x batch x context_state_size
        # mask has shape seqLen x batch

        scores = matmul3d(hidden, self.hidden_to_score) # seqLen x batch x 1
        scores = tf.squeeze(scores, axis=2)
        scores = scores - tf.reduce_max(scores, axis=0, keep_dims=True)
        scores = tf.exp(scores)
        scores *= self.context_mask
        scores = scores / tf.reduce_sum(scores, axis=0, keep_dims=True)

        attention_context = self.context * tf.expand_dims(scores, axis=2)
        attention_context = tf.reduce_sum(attention_context, axis=0, keep_dims=False)

        return attention_context

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
        cost = tf.reduce_sum(cost, axis=0, keep_dims=False)
        return cost
