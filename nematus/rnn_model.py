import tensorflow as tf

import layers
import model_inputs

from sampling_utils import SamplingUtils


"""Builds a GRU-based attentional encoder-decoder model.

This class is responsible for constructing a TensorFlow graph that takes
a minibatch of sentence pairs as input and calculates a loss value via
teacher forcing (specifically, the loss is the mean sentence-level
cross-entropy).

For inference (sampling and beam search), see rnn_inference.py.

For optimization, see model_updater.py.
"""
class RNNModel(object):
    def __init__(self, config):
        self.inputs = model_inputs.ModelInputs(config)

        # Dropout functions for words.
        # These probabilistically zero-out all embedding values for individual
        # words.
        dropout_source, dropout_target = None, None
        if config.rnn_use_dropout and config.rnn_dropout_source > 0.0:
            def dropout_source(x):
                return tf.layers.dropout(
                    x, noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1),
                    rate=config.rnn_dropout_source,
                    training=self.inputs.training)
        if config.rnn_use_dropout and config.rnn_dropout_target > 0.0:
            def dropout_target(y):
                return tf.layers.dropout(
                    y, noise_shape=(tf.shape(y)[0], tf.shape(y)[1], 1),
                    rate=config.rnn_dropout_target,
                    training=self.inputs.training)

        # Dropout functions for use within FF, GRU, and attention layers.
        # We use Gal and Ghahramani (2016)-style dropout, so these functions
        # will be used to create 2D dropout masks that are reused at every
        # timestep.
        dropout_embedding, dropout_hidden = None, None
        if config.rnn_use_dropout and config.rnn_dropout_embedding > 0.0:
            def dropout_embedding(e):
                return tf.layers.dropout(e, noise_shape=tf.shape(e),
                                         rate=config.rnn_dropout_embedding,
                                         training=self.inputs.training)
        if config.rnn_use_dropout and config.rnn_dropout_hidden > 0.0:
            def dropout_hidden(h):
                return tf.layers.dropout(h, noise_shape=tf.shape(h),
                                         rate=config.rnn_dropout_hidden,
                                         training=self.inputs.training)

        batch_size = tf.shape(self.inputs.x)[-1]  # dynamic value

        with tf.variable_scope("encoder"):
            self.encoder = Encoder(config, batch_size, dropout_source,
                                   dropout_embedding, dropout_hidden)
            ctx, embs = self.encoder.get_context(self.inputs.x, self.inputs.x_mask)

        with tf.variable_scope("decoder"):
            if config.tie_encoder_decoder_embeddings:
                tied_embeddings = self.encoder.emb_layer
            else:
                tied_embeddings = None
            self.decoder = Decoder(config, ctx, embs, self.inputs.x_mask,
                                   dropout_target, dropout_embedding,
                                   dropout_hidden, tied_embeddings)
            self.logits = self.decoder.score(self.inputs.y)

        with tf.variable_scope("loss"):
            self.loss_layer = layers.Masked_cross_entropy_loss(
                self.inputs.y, self.inputs.y_mask, config.label_smoothing,
                training=self.inputs.training)
            self._loss_per_sentence = self.loss_layer.forward(self.logits)
            self._loss = tf.reduce_mean(self._loss_per_sentence, keepdims=False)

        self.sampling_utils = SamplingUtils(config)

    @property
    def loss_per_sentence(self):
        return self._loss_per_sentence

    @property
    def loss(self):
        return self._loss


class Decoder(object):
    def __init__(self, config, context, x_embs, x_mask,
                 dropout_target, dropout_embedding, dropout_hidden,
                 encoder_embedding_layer=None):

        self.dropout_target = dropout_target
        batch_size = tf.shape(x_mask)[1]

        with tf.variable_scope("initial_state_constructor"):
            context_sum = tf.reduce_sum(
                            context * tf.expand_dims(x_mask, axis=2),
                            axis=0)

            context_mean = context_sum / tf.expand_dims(
                                            tf.reduce_sum(x_mask, axis=0),
                                            axis=1)
            self.init_state_layer = layers.FeedForwardLayer(
                in_size=config.state_size * 2,
                out_size=config.state_size,
                batch_size=batch_size,
                use_layer_norm=config.rnn_layer_normalization,
                dropout_input=dropout_hidden)
            self.init_state = self.init_state_layer.forward(context_mean)
            self.x_embs = x_embs

            self.translation_maxlen = config.translation_maxlen
            self.embedding_size = config.target_embedding_size
            self.state_size = config.state_size
            self.target_vocab_size = config.target_vocab_size

        with tf.variable_scope("embedding"):
            if encoder_embedding_layer == None:
                self.y_emb_layer = layers.EmbeddingLayer(
                    vocabulary_sizes=[config.target_vocab_size],
                    dim_per_factor=[config.target_embedding_size])
            else:
                self.y_emb_layer = encoder_embedding_layer

        with tf.variable_scope("base"):
            with tf.variable_scope("gru0"):
                if config.theano_compat:
                    bias_type = layers.LegacyBiasType.THEANO_A
                else:
                    bias_type = layers.LegacyBiasType.NEMATUS_COMPAT_FALSE
                self.grustep1 = layers.GRUStep(
                    input_size=config.target_embedding_size,
                    state_size=config.state_size,
                    batch_size=batch_size,
                    use_layer_norm=config.rnn_layer_normalization,
                    legacy_bias_type=bias_type,
                    dropout_input=dropout_embedding,
                    dropout_state=dropout_hidden)
            with tf.variable_scope("attention"):
                self.attstep = layers.AttentionStep(
                    context=context,
                    context_state_size=2*config.state_size,
                    context_mask=x_mask,
                    state_size=config.state_size,
                    hidden_size=2*config.state_size,
                    use_layer_norm=config.rnn_layer_normalization,
                    dropout_context=dropout_hidden,
                    dropout_state=dropout_hidden)
            if config.theano_compat:
                bias_type = layers.LegacyBiasType.THEANO_B
            else:
                bias_type = layers.LegacyBiasType.NEMATUS_COMPAT_TRUE
            self.grustep2 = layers.DeepTransitionGRUStep(
                input_size=2*config.state_size,
                state_size=config.state_size,
                batch_size=batch_size,
                use_layer_norm=config.rnn_layer_normalization,
                legacy_bias_type=bias_type,
                dropout_input=dropout_hidden,
                dropout_state=dropout_hidden,
                transition_depth=config.rnn_dec_base_transition_depth-1,
                var_scope_fn=lambda i: "gru{0}".format(i+1))

        with tf.variable_scope("high"):
            if config.rnn_dec_depth == 1:
                self.high_gru_stack = None
            else:
                if config.theano_compat:
                    bias_type = layers.LegacyBiasType.THEANO_A
                else:
                    bias_type = layers.LegacyBiasType.NEMATUS_COMPAT_TRUE
                self.high_gru_stack = layers.GRUStack(
                    input_size=config.state_size,
                    state_size=config.state_size,
                    batch_size=batch_size,
                    use_layer_norm=config.rnn_layer_normalization,
                    legacy_bias_type=bias_type,
                    dropout_input=dropout_hidden,
                    dropout_state=dropout_hidden,
                    stack_depth=config.rnn_dec_depth-1,
                    transition_depth=config.rnn_dec_high_transition_depth,
                    context_state_size=(2*config.state_size if config.rnn_dec_deep_context else 0),
                    residual_connections=True,
                    first_residual_output=0)

        if config.rnn_lexical_model:
            with tf.variable_scope("lexical"):
                self.lexical_layer = layers.LexicalModel(
                    in_size=config.embedding_size,
                    out_size=config.embedding_size,
                    batch_size=batch_size,
                    use_layer_norm=config.rnn_layer_normalization,
                    dropout_embedding=dropout_embedding,
                    dropout_hidden=dropout_hidden)
        else:
            self.lexical_layer = None

        with tf.variable_scope("next_word_predictor"):
            W = None
            if config.tie_decoder_embeddings:
                W = self.y_emb_layer.get_embeddings(factor=0)
                W = tf.transpose(W)
            self.predictor = Predictor(config, batch_size, dropout_embedding,
                                       dropout_hidden, hidden_to_logits_W=W)


    def score(self, y):
        with tf.variable_scope("y_embeddings_layer"):
            y_but_last = tf.slice(y, [0,0], [tf.shape(y)[0]-1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last, factor=0)
            if self.dropout_target != None:
                y_embs = self.dropout_target(y_embs)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        init_attended_context = tf.zeros([tf.shape(self.init_state)[0], self.state_size*2])
        init_att_alphas = tf.zeros([tf.shape(self.x_embs)[0], tf.shape(self.x_embs)[1]])
        init_state_att_ctx = (self.init_state, init_attended_context, init_att_alphas)
        gates_x, proposal_x = self.grustep1.precompute_from_x(y_embs)
        def step_fn(prev, x):
            prev_state = prev[0]
            prev_att_ctx = prev[1]
            prev_lexical_state = prev[2]
            gates_x2d = x[0]
            proposal_x2d = x[1]
            state = self.grustep1.forward(
                        prev_state,
                        gates_x=gates_x2d,
                        proposal_x=proposal_x2d)
            att_ctx, att_alphas = self.attstep.forward(state) 
            state = self.grustep2.forward(state, att_ctx)
            #TODO: write att_ctx to tensorArray instead of having it as output of scan?
            return (state, att_ctx, att_alphas)

        layer = layers.RecurrentLayer(initial_state=init_state_att_ctx,
                                      step_fn=step_fn)
        states, attended_states, attention_weights = layer.forward((gates_x, proposal_x))

        if self.high_gru_stack != None:
            states = self.high_gru_stack.forward(
                states,
                context_layer=(attended_states if self.high_gru_stack.context_state_size > 0 else None),
                init_state=self.init_state)

        if self.lexical_layer is not None:
            lexical_states = self.lexical_layer.forward(self.x_embs, attention_weights, multi_step=True)
        else:
            lexical_states = None

        logits = self.predictor.get_logits(y_embs, states, attended_states, lexical_states, multi_step=True)
        return logits

class Predictor(object):
    def __init__(self, config, batch_size, dropout_embedding, dropout_hidden, hidden_to_logits_W=None):
        self.config = config

        with tf.variable_scope("prev_emb_to_hidden"):
            self.prev_emb_to_hidden = layers.FeedForwardLayer(
                in_size=config.target_embedding_size,
                out_size=config.target_embedding_size,
                batch_size=batch_size,
                non_linearity=lambda y: y,
                use_layer_norm=config.rnn_layer_normalization,
                dropout_input=dropout_embedding)
        with tf.variable_scope("state_to_hidden"):
            self.state_to_hidden = layers.FeedForwardLayer(
                in_size=config.state_size,
                out_size=config.target_embedding_size,
                batch_size=batch_size,
                non_linearity=lambda y: y,
                use_layer_norm=config.rnn_layer_normalization,
                dropout_input=dropout_hidden)
        with tf.variable_scope("attended_context_to_hidden"):
            self.att_ctx_to_hidden = layers.FeedForwardLayer(
                in_size=2*config.state_size,
                out_size=config.target_embedding_size,
                batch_size=batch_size,
                non_linearity=lambda y: y,
                use_layer_norm=config.rnn_layer_normalization,
                dropout_input=dropout_hidden)

        if config.output_hidden_activation == 'prelu':
            with tf.variable_scope("hidden_prelu"):
                self.hidden_prelu = PReLU(in_size=config.target_embedding_size)

        with tf.variable_scope("hidden_to_logits"):
            self.hidden_to_logits = layers.FeedForwardLayer(
                in_size=config.target_embedding_size,
                out_size=config.target_vocab_size,
                batch_size=batch_size,
                non_linearity=lambda y: y,
                W=hidden_to_logits_W,
                dropout_input=dropout_embedding)

        if config.softmax_mixture_size > 1:
            with tf.variable_scope("hidden_to_pi_logits"):
                self.hidden_to_pi_logits = layers.FeedForwardLayer(
                    in_size=config.target_embedding_size,
                    out_size=config.softmax_mixture_size,
                    batch_size=batch_size,
                    non_linearity=lambda y: y,
                    dropout_input=dropout_embedding)
            self.hidden_to_mos_hidden = []
            for k in range(config.softmax_mixture_size):
                with tf.variable_scope("hidden_to_mos_hidden_{}".format(k)):
                    layer = layers.FeedForwardLayer(
                        in_size=config.target_embedding_size,
                        out_size=config.target_embedding_size,
                        batch_size=batch_size,
                        use_layer_norm=config.rnn_layer_normalization,
                        dropout_input=dropout_embedding)
                    self.hidden_to_mos_hidden.append(layer)

        if config.rnn_lexical_model:
            with tf.variable_scope("lexical_to_logits"):
                self.lexical_to_logits = layers.FeedForwardLayer(
                                in_size=config.target_embedding_size,
                                out_size=config.target_vocab_size,
                                batch_size=batch_size,
                                non_linearity=lambda y: y,
                                dropout_input=dropout_embedding)

    def get_logits(self, y_embs, states, attended_states, lexical_states, multi_step=True):
        with tf.variable_scope("prev_emb_to_hidden"):
            hidden_emb = self.prev_emb_to_hidden.forward(y_embs, input_is_3d=multi_step)

        with tf.variable_scope("state_to_hidden"):
            hidden_state = self.state_to_hidden.forward(states, input_is_3d=multi_step)

        with tf.variable_scope("attended_context_to_hidden"):
            hidden_att_ctx = self.att_ctx_to_hidden.forward(attended_states,input_is_3d=multi_step)

        hidden = hidden_emb + hidden_state + hidden_att_ctx
        if self.config.output_hidden_activation == 'tanh':
            hidden = tf.tanh(hidden)
        elif self.config.output_hidden_activation == 'relu':
            hidden = tf.nn.relu(hidden)
        elif self.config.output_hidden_activation == 'prelu':
            hidden = self.hidden_prelu.forward(hidden)
        elif self.config.output_hidden_activation == 'linear':
            pass
        else:
            assert False, 'Unknown output activation function "%s"' % self.config.output_hidden_activation

        if self.config.softmax_mixture_size == 1:
            with tf.variable_scope("hidden_to_logits"):
                logits = self.hidden_to_logits.forward(hidden, input_is_3d=multi_step)

            if self.config.rnn_lexical_model:
                with tf.variable_scope("lexical_to_logits"):
                    logits += self.lexical_to_logits.forward(lexical_states, input_is_3d=multi_step)

        else:
            assert self.config.softmax_mixture_size > 1
            pi_logits = self.hidden_to_pi_logits.forward(hidden,
                                                         input_is_3d=multi_step)
            pi = tf.nn.softmax(pi_logits)
            probs = None
            for k in range(self.config.softmax_mixture_size):
                hidden_k = self.hidden_to_mos_hidden[k].forward(hidden,
                    input_is_3d=multi_step)
                logits_k = self.hidden_to_logits.forward(hidden_k,
                    input_is_3d=multi_step)
                probs_k = tf.nn.softmax(logits_k)
                weight = pi[..., k:k+1]
                if k == 0:
                    probs = probs_k * weight
                else:
                    probs += probs_k * weight
            logits = tf.log(probs)

        return logits 


class Encoder(object):
    def __init__(self, config, batch_size, dropout_source, dropout_embedding,
                 dropout_hidden):

        self.dropout_source = dropout_source

        with tf.variable_scope("embedding"):
            self.emb_layer = layers.EmbeddingLayer(config.source_vocab_sizes,
                                                   config.dim_per_factor)

        if config.theano_compat:
            bias_type = layers.LegacyBiasType.THEANO_A
        else:
            bias_type = layers.LegacyBiasType.NEMATUS_COMPAT_FALSE

        with tf.variable_scope("forward-stack"):
            self.forward_encoder = layers.GRUStack(
                input_size=config.embedding_size,
                state_size=config.state_size,
                batch_size=batch_size,
                use_layer_norm=config.rnn_layer_normalization,
                legacy_bias_type=bias_type,
                dropout_input=dropout_embedding,
                dropout_state=dropout_hidden,
                stack_depth=config.rnn_enc_depth,
                transition_depth=config.rnn_enc_transition_depth,
                alternating=True,
                residual_connections=True,
                first_residual_output=1)

        with tf.variable_scope("backward-stack"):
            self.backward_encoder = layers.GRUStack(
                input_size=config.embedding_size,
                state_size=config.state_size,
                batch_size=batch_size,
                use_layer_norm=config.rnn_layer_normalization,
                legacy_bias_type=bias_type,
                dropout_input=dropout_embedding,
                dropout_state=dropout_hidden,
                stack_depth=config.rnn_enc_depth,
                transition_depth=config.rnn_enc_transition_depth,
                alternating=True,
                reverse_alternation=True,
                residual_connections=True,
                first_residual_output=1)

    def get_context(self, x, x_mask):

        with tf.variable_scope("embedding"):
            embs = self.emb_layer.forward(x)
            if self.dropout_source != None:
                embs = self.dropout_source(embs)

        with tf.variable_scope("forward-stack"):
            fwd_states = self.forward_encoder.forward(embs, x_mask)

        with tf.variable_scope("backward-stack"):
            bwd_states = self.backward_encoder.forward(embs, x_mask)

        # Concatenate the left-to-right and the right-to-left states, in that
        # order. This is for compatibility with models that were trained with
        # the Theano version.
        stack_depth = len(self.forward_encoder.grus)
        if stack_depth % 2 == 0:
            concat_states = tf.concat([bwd_states, fwd_states], axis=2)
        else:
            concat_states = tf.concat([fwd_states, bwd_states], axis=2)
        return concat_states, embs
