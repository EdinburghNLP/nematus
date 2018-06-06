import sys
import logging

import numpy

import tensorflow as tf
from layers import *
import inference


class Decoder(object):
    def __init__(self, config, context, x_mask, dropout_target,
                 dropout_embedding, dropout_hidden):

        self.dropout_target = dropout_target
        batch_size = tf.shape(x_mask)[1]

        with tf.name_scope("initial_state_constructor"):
            context_sum = tf.reduce_sum(
                            context * tf.expand_dims(x_mask, axis=2),
                            axis=0)

            context_mean = context_sum / tf.expand_dims(
                                            tf.reduce_sum(x_mask, axis=0),
                                            axis=1)
            self.init_state_layer = FeedForwardLayer(
                                        in_size=config.state_size * 2,
                                        out_size=config.state_size,
                                        batch_size=batch_size,
                                        use_layer_norm=config.use_layer_norm,
                                        dropout_input=dropout_hidden)
            self.init_state = self.init_state_layer.forward(context_mean)

            self.translation_maxlen = config.translation_maxlen
            self.embedding_size = config.embedding_size
            self.state_size = config.state_size
            self.target_vocab_size = config.target_vocab_size

        with tf.name_scope("embedding"):
            self.y_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.target_vocab_size,
                                embedding_size=config.embedding_size)

        with tf.name_scope("base"):
            with tf.name_scope("gru0"):
                self.grustep1 = GRUStep(
                                    input_size=config.embedding_size,
                                    state_size=config.state_size,
                                    batch_size=batch_size,
                                    use_layer_norm=config.use_layer_norm,
                                    nematus_compat=False,
                                    dropout_input=dropout_embedding,
                                    dropout_state=dropout_hidden)
            with tf.name_scope("attention"):
                self.attstep = AttentionStep(
                                context=context,
                                context_state_size=2*config.state_size,
                                context_mask=x_mask,
                                state_size=config.state_size,
                                hidden_size=2*config.state_size,
                                use_layer_norm=config.use_layer_norm,
                                dropout_context=dropout_hidden,
                                dropout_state=dropout_hidden)
            self.grustep2 = DeepTransitionGRUStep(
                                    input_size=2*config.state_size,
                                    state_size=config.state_size,
                                    batch_size=batch_size,
                                    use_layer_norm=config.use_layer_norm,
                                    nematus_compat=True,
                                    dropout_input=dropout_hidden,
                                    dropout_state=dropout_hidden,
                                    transition_depth=config.dec_base_recurrence_transition_depth-1,
                                    name_scope_fn=lambda i: "gru{0}".format(i+1))

        with tf.name_scope("high"):
            if config.dec_depth == 1:
                self.high_gru_stack = None
            else:
                self.high_gru_stack = GRUStack(
                    input_size=config.state_size,
                    state_size=config.state_size,
                    batch_size=batch_size,
                    use_layer_norm=config.use_layer_norm,
                    nematus_compat=True,
                    dropout_input=dropout_hidden,
                    dropout_state=dropout_hidden,
                    stack_depth=config.dec_depth-1,
                    transition_depth=config.dec_high_recurrence_transition_depth,
                    context_state_size=(2*config.state_size if config.dec_deep_context else 0),
                    residual_connections=True,
                    first_residual_output=0)

        with tf.name_scope("next_word_predictor"):
            W = None
            if config.tie_decoder_embeddings:
                W = self.y_emb_layer.get_embeddings()
                W = tf.transpose(W)
            self.predictor = Predictor(config, batch_size, dropout_embedding,
                                       dropout_hidden, hidden_to_logits_W=W)


    def sample(self):
        batch_size = tf.shape(self.init_state)[0]
        high_depth = 0 if self.high_gru_stack == None \
                       else len(self.high_gru_stack.grus)
        i = tf.constant(0)
        init_y = -tf.ones(dtype=tf.int32, shape=[batch_size])
        init_emb = tf.zeros(dtype=tf.float32,
                            shape=[batch_size,self.embedding_size])
        y_array = tf.TensorArray(
            dtype=tf.int32,
            size=self.translation_maxlen,
            clear_after_read=True, #TODO: does this help? or will it only introduce bugs in the future?
            name='y_sampled_array')
        init_loop_vars = [i, self.init_state, [self.init_state] * high_depth,
                          init_y, init_emb, y_array]

        def cond(i, base_state, high_states, prev_y, prev_emb, y_array):
            return tf.logical_and(
                tf.less(i, self.translation_maxlen),
                tf.reduce_any(tf.not_equal(prev_y, 0)))

        def body(i, prev_base_state, prev_high_states, prev_y, prev_emb,
                 y_array):
            state1 = self.grustep1.forward(prev_base_state, prev_emb)
            att_ctx = self.attstep.forward(state1)
            base_state = self.grustep2.forward(state1, att_ctx)
            if self.high_gru_stack == None:
                output = base_state
                high_states = []
            else:
                if self.high_gru_stack.context_state_size == 0:
                    output, high_states = self.high_gru_stack.forward_single(
                        prev_high_states, base_state)
                else:
                    output, high_states = self.high_gru_stack.forward_single(
                        prev_high_states, base_state, context=att_ctx)
            logits = self.predictor.get_logits(prev_emb, output, att_ctx,
                                               multi_step=False)
            new_y = tf.multinomial(logits, num_samples=1)
            new_y = tf.cast(new_y, dtype=tf.int32)
            new_y = tf.squeeze(new_y, axis=1)
            new_y = tf.where(tf.equal(prev_y, tf.constant(0, dtype=tf.int32)),
                             tf.zeros_like(new_y), new_y)
            y_array = y_array.write(index=i, value=new_y)
            new_emb = self.y_emb_layer.forward(new_y)
            return i+1, base_state, high_states, new_y, new_emb, y_array

        final_loop_vars = tf.while_loop(
                           cond=cond,
                           body=body,
                           loop_vars=init_loop_vars,
                           back_prop=False)
        i, _, _, _, _, y_array = final_loop_vars
        sampled_ys = y_array.gather(tf.range(0, i))
        return sampled_ys

    def score(self, y):
        with tf.name_scope("y_embeddings_layer"):
            y_but_last = tf.slice(y, [0,0], [tf.shape(y)[0]-1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last)
            if self.dropout_target != None:
                y_embs = self.dropout_target(y_embs)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        init_attended_context = tf.zeros([tf.shape(self.init_state)[0], self.state_size*2])
        init_state_att_ctx = (self.init_state, init_attended_context)
        gates_x, proposal_x = self.grustep1.precompute_from_x(y_embs)
        def step_fn(prev, x):
            prev_state = prev[0]
            prev_att_ctx = prev[1]
            gates_x2d = x[0]
            proposal_x2d = x[1]
            state = self.grustep1.forward(
                        prev_state,
                        gates_x=gates_x2d,
                        proposal_x=proposal_x2d)
            att_ctx = self.attstep.forward(state) 
            state = self.grustep2.forward(state, att_ctx)
            #TODO: write att_ctx to tensorArray instead of having it as output of scan?
            return (state, att_ctx)

        states, attended_states = RecurrentLayer(
                                    initial_state=init_state_att_ctx,
                                    step_fn=step_fn).forward((gates_x, proposal_x))

        if self.high_gru_stack != None:
            states = self.high_gru_stack.forward(
                states,
                context_layer=(attended_states if self.high_gru_stack.context_state_size > 0 else None))

        logits = self.predictor.get_logits(y_embs, states, attended_states, multi_step=True)
        return logits

class Predictor(object):
    def __init__(self, config, batch_size, dropout_embedding, dropout_hidden, hidden_to_logits_W=None):
        self.config = config

        with tf.name_scope("prev_emb_to_hidden"):
            self.prev_emb_to_hidden = FeedForwardLayer(
                                in_size=config.embedding_size,
                                out_size=config.embedding_size,
                                batch_size=batch_size,
                                non_linearity=lambda y: y,
                                use_layer_norm=config.use_layer_norm,
                                dropout_input=dropout_embedding)
        with tf.name_scope("state_to_hidden"):
            self.state_to_hidden = FeedForwardLayer(
                                    in_size=config.state_size,
                                    out_size=config.embedding_size,
                                    batch_size=batch_size,
                                    non_linearity=lambda y: y,
                                    use_layer_norm=config.use_layer_norm,
                                    dropout_input=dropout_hidden)
        with tf.name_scope("attended_context_to_hidden"):
            self.att_ctx_to_hidden = FeedForwardLayer(
                                    in_size=2*config.state_size,
                                    out_size=config.embedding_size,
                                    batch_size=batch_size,
                                    non_linearity=lambda y: y,
                                    use_layer_norm=config.use_layer_norm,
                                    dropout_input=dropout_hidden)

        if config.output_hidden_activation == 'prelu':
            with tf.name_scope("hidden_prelu"):
                self.hidden_prelu = PReLU(in_size=config.embedding_size)

        with tf.name_scope("hidden_to_logits"):
            self.hidden_to_logits = FeedForwardLayer(
                            in_size=config.embedding_size,
                            out_size=config.target_vocab_size,
                            batch_size=batch_size,
                            non_linearity=lambda y: y,
                            W=hidden_to_logits_W,
                            dropout_input=dropout_embedding)

    def get_logits(self, y_embs, states, attended_states, multi_step=True):
        with tf.name_scope("prev_emb_to_hidden"):
            hidden_emb = self.prev_emb_to_hidden.forward(y_embs, input_is_3d=multi_step)

        with tf.name_scope("state_to_hidden"):
            hidden_state = self.state_to_hidden.forward(states, input_is_3d=multi_step)

        with tf.name_scope("attended_context_to_hidden"):
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

        with tf.name_scope("hidden_to_logits"):
            logits = self.hidden_to_logits.forward(hidden, input_is_3d=multi_step)
        
        return logits 


class Encoder(object):
    def __init__(self, config, batch_size, dropout_source, dropout_embedding,
                 dropout_hidden):

        self.dropout_source = dropout_source

        with tf.name_scope("embedding"):
            self.emb_layer = EmbeddingLayerWithFactors(
                config.source_vocab_sizes,
                config.dim_per_factor)

        with tf.name_scope("forward-stack"):
            self.forward_encoder = GRUStack(
                    input_size=config.embedding_size,
                    state_size=config.state_size,
                    batch_size=batch_size,
                    use_layer_norm=config.use_layer_norm,
                    nematus_compat=False,
                    dropout_input=dropout_embedding,
                    dropout_state=dropout_hidden,
                    stack_depth=config.enc_depth,
                    transition_depth=config.enc_recurrence_transition_depth,
                    alternating=True,
                    residual_connections=True,
                    first_residual_output=1)

        with tf.name_scope("backward-stack"):
            self.backward_encoder = GRUStack(
                    input_size=config.embedding_size,
                    state_size=config.state_size,
                    batch_size=batch_size,
                    use_layer_norm=config.use_layer_norm,
                    nematus_compat=False,
                    dropout_input=dropout_embedding,
                    dropout_state=dropout_hidden,
                    stack_depth=config.enc_depth,
                    transition_depth=config.enc_recurrence_transition_depth,
                    alternating=True,
                    reverse_alternation=True,
                    residual_connections=True,
                    first_residual_output=1)

    def get_context(self, x, x_mask):

        with tf.name_scope("embedding"):
            embs = self.emb_layer.forward(x)
            if self.dropout_source != None:
                embs = self.dropout_source(embs)

        with tf.name_scope("forward-stack"):
            fwd_states = self.forward_encoder.forward(embs, x_mask)

        with tf.name_scope("backward-stack"):
            bwd_states = self.backward_encoder.forward(embs, x_mask)

        # Concatenate the left-to-right and the right-to-left states, in that
        # order. This is for compatibility with models that were trained with
        # the Theano version.
        stack_depth = len(self.forward_encoder.grus)
        if stack_depth % 2 == 0:
            concat_states = tf.concat([bwd_states, fwd_states], axis=2)
        else:
            concat_states = tf.concat([fwd_states, bwd_states], axis=2)
        return concat_states


class StandardModel(object):
    def __init__(self, config):

        #variable dimensions
        seqLen = None
        batch_size = None

        self.x = tf.placeholder(
                    dtype=tf.int32,
                    name='x',
                    shape=(config.factors, seqLen, batch_size))
        self.x_mask = tf.placeholder(
                        dtype=tf.float32,
                        name='x_mask',
                        shape=(seqLen, batch_size))
        self.y = tf.placeholder(
                    dtype=tf.int32,
                    name='y',
                    shape=(seqLen, batch_size))
        self.y_mask = tf.placeholder(
                        dtype=tf.float32,
                        name='y_mask',
                        shape=(seqLen, batch_size))
        self.training = tf.placeholder_with_default(
            False, name='training', shape=())

        # Dropout functions for words.
        # These probabilistically zero-out all embedding values for individual
        # words.
        dropout_source, dropout_target = None, None
        if config.use_dropout and config.dropout_source > 0.0:
            def dropout_source(x):
                return tf.layers.dropout(
                    x, noise_shape=(tf.shape(x)[0], tf.shape(x)[1], 1),
                    rate=config.dropout_source, training=self.training)
        if config.use_dropout and config.dropout_target > 0.0:
            def dropout_target(y):
                return tf.layers.dropout(
                    y, noise_shape=(tf.shape(y)[0], tf.shape(y)[1], 1),
                    rate=config.dropout_target, training=self.training)

        # Dropout functions for use within FF, GRU, and attention layers.
        # We use Gal and Ghahramani (2016)-style dropout, so these functions
        # will be used to create 2D dropout masks that are reused at every
        # timestep.
        dropout_embedding, dropout_hidden = None, None
        if config.use_dropout and config.dropout_embedding > 0.0:
            def dropout_embedding(e):
                return tf.layers.dropout(e, noise_shape=tf.shape(e),
                                         rate=config.dropout_embedding,
                                         training=self.training)
        if config.use_dropout and config.dropout_hidden > 0.0:
            def dropout_hidden(h):
                return tf.layers.dropout(h, noise_shape=tf.shape(h),
                                         rate=config.dropout_hidden,
                                         training=self.training)

        batch_size = tf.shape(self.x)[-1]  # dynamic value

        with tf.name_scope("encoder"):
            self.encoder = Encoder(config, batch_size, dropout_source,
                                   dropout_embedding, dropout_hidden)
            ctx = self.encoder.get_context(self.x, self.x_mask)

        with tf.name_scope("decoder"):
            self.decoder = Decoder(config, ctx, self.x_mask, dropout_target,
                                   dropout_embedding, dropout_hidden)
            self.logits = self.decoder.score(self.y)

        with tf.name_scope("loss"):
            self.loss_layer = Masked_cross_entropy_loss(self.y, self.y_mask)
            self.loss_per_sentence = self.loss_layer.forward(self.logits)
            self.mean_loss = tf.reduce_mean(self.loss_per_sentence, keep_dims=False)
            self.objective = self.mean_loss
            
            self.l2_loss = tf.constant(0.0, dtype=tf.float32)
            if config.decay_c > 0.0:
                self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * tf.constant(config.decay_c, dtype=tf.float32)
                self.objective += self.l2_loss

            self.map_l2_loss = tf.constant(0.0, dtype=tf.float32)
            if config.map_decay_c > 0.0:
                map_l2_acc = []
                for v in tf.trainable_variables():
                    prior_name = 'prior/'+v.name.split(':')[0]
                    prior_v = tf.Variable(initial_value=v.initialized_value(), trainable=False, collections=['prior_variables'], name=prior_name, dtype=v.initialized_value().dtype)
                    map_l2_acc.append(tf.nn.l2_loss(v - prior_v))
                self.map_l2_loss = tf.add_n(map_l2_acc) * tf.constant(config.map_decay_c, dtype=tf.float32)
                self.objective += self.l2_loss

        if config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        else:
            logging.error('No valid optimizer defined: {0}'.format(config.optimizer))
            sys.exit(1)

        self.t = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
        grad_vars = self.optimizer.compute_gradients(self.mean_loss)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
        # Might be interesting to see how the global norm changes over time, attach a summary?
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = self.optimizer.apply_gradients(grad_vars, global_step=self.t)

        self.sampled_ys = None
        self.beam_size, self.beam_ys, self.parents, self.cost = None, None, None, None

    def get_score_inputs(self):
        return self.x, self.x_mask, self.y, self.y_mask, self.training
    
    def get_loss(self):
        return self.loss_per_sentence

    def get_mean_loss(self):
        return self.mean_loss

    def get_objective(self):
        return self.objective

    def get_global_step(self):
        return self.t

    def reset_global_step(self, value, session):
        self.t.load(value, session)

    def get_apply_grads(self):
        return self.apply_grads

    def _get_samples(self):
        if self.sampled_ys == None:
            self.sampled_ys = self.decoder.sample()
        return self.sampled_ys

    def sample(self, session, x_in, x_mask_in):
        sampled_ys = self._get_samples()
        feeds = {self.x : x_in, self.x_mask : x_mask_in}
        sampled_ys_out = session.run(sampled_ys, feed_dict=feeds)
        sampled_ys_out = sampled_ys_out.T
        samples = []
        for sample in sampled_ys_out:
            sample = numpy.trim_zeros(list(sample), trim='b')
            sample.append(0)
            samples.append(sample)
        return samples



    def _get_beam_search_outputs(self, beam_size):
        if beam_size != self.beam_size:
            self.beam_size = beam_size
            self.beam_ys, self.parents, self.cost = inference.construct_beam_search_functions([self], beam_size)
        return self.beam_ys, self.parents, self.cost


    def beam_search(self, session, x_in, x_mask_in, beam_size):
        # x_in is a numpy array with shape (factors, seqLen, batch)
        # x_mask is a numpy array with shape (seqLen, batch)
        x_in = numpy.repeat(x_in, repeats=beam_size, axis=-1)
        x_mask_in = numpy.repeat(x_mask_in, repeats=beam_size, axis=-1)
        feeds = {self.x : x_in, self.x_mask : x_mask_in}
        beam_ys, parents, cost = self._get_beam_search_outputs(beam_size)
        beam_ys_out, parents_out, cost_out = session.run(
                                                    [beam_ys, parents, cost],
                                                    feed_dict=feeds)
        return inference.reconstruct_hypotheses(beam_ys_out, parents_out, cost_out, beam_size)
