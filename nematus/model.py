import sys
import logging

import numpy

import tensorflow as tf
from layers import *
import inference


class Decoder(object):
    def __init__(self, config, context, x_mask):
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
                                        use_layer_norm=config.use_layer_norm)
            self.init_state = self.init_state_layer.forward(context_mean)

            self.translation_maxlen = config.translation_maxlen
            self.embedding_size = config.embedding_size
            self.state_size = config.state_size
            self.target_vocab_size = config.target_vocab_size

        with tf.name_scope("y_embeddings_layer"):
            self.y_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.target_vocab_size,
                                embedding_size=config.embedding_size)

        self.grustep1 = GRUStep(
                            input_size=config.embedding_size,
                            state_size=config.state_size,
                            use_layer_norm=config.use_layer_norm,
                            nematus_compat=False)
        self.attstep = AttentionStep(
                        context=context,
                        context_state_size=2*config.state_size,
                        context_mask=x_mask,
                        state_size=config.state_size,
                        hidden_size=2*config.state_size,
                        use_layer_norm=config.use_layer_norm)
        self.grustep2 = GRUStep(
                            input_size=2*config.state_size,
                            state_size=config.state_size,
                            use_layer_norm=config.use_layer_norm,
                            nematus_compat=True)
        with tf.name_scope("next_word_predictor"):
            W = None
            if config.tie_decoder_embeddings:
                W = self.y_emb_layer.get_embeddings()
                W = tf.transpose(W)
            self.predictor = Predictor(
                                config,
                                hidden_to_logits_W=W)


    def sample(self):
       batch_size = tf.shape(self.init_state)[0]
       i = tf.constant(0)
       init_ys = -tf.ones(dtype=tf.int32, shape=[batch_size])
       init_embs = tf.zeros(dtype=tf.float32, shape=[batch_size,self.embedding_size])
       ys_array = tf.TensorArray(
                    dtype=tf.int32,
                    size=self.translation_maxlen,
                    clear_after_read=True, #TODO: does this help? or will it only introduce bugs in the future?
                    name='y_sampled_array')
       init_loop_vars = [i, self.init_state, init_ys, init_embs, ys_array]
       def cond(i, states, prev_ys, prev_embs, ys_array):
           return tf.logical_and(
                   tf.less(i, self.translation_maxlen),
                   tf.reduce_any(tf.not_equal(prev_ys, 0)))

       def body(i, states, prev_ys, prev_embs, ys_array):
           new_states1 = self.grustep1.forward(states, prev_embs)
           att_ctx = self.attstep.forward(new_states1)
           new_states2 = self.grustep2.forward(new_states1, att_ctx)
           logits = self.predictor.get_logits(prev_embs, new_states2, att_ctx, multi_step=False)
           new_ys = tf.multinomial(logits, num_samples=1)
           new_ys = tf.cast(new_ys, dtype=tf.int32)
           new_ys = tf.squeeze(new_ys, axis=1)
           new_ys = tf.where(
                   tf.equal(prev_ys, tf.constant(0, dtype=tf.int32)),
                   tf.zeros_like(new_ys),
                   new_ys)
           ys_array = ys_array.write(index=i, value=new_ys)
           new_embs = self.y_emb_layer.forward(new_ys)
           return i+1, new_states2, new_ys, new_embs, ys_array

       final_loop_vars = tf.while_loop(
                           cond=cond,
                           body=body,
                           loop_vars=init_loop_vars,
                           back_prop=False)
       i, _, _, _, ys_array = final_loop_vars
       sampled_ys = ys_array.gather(tf.range(0, i))
       return sampled_ys

    def score(self, y):
        with tf.name_scope("y_embeddings_layer"):
            y_but_last = tf.slice(y, [0,0], [tf.shape(y)[0]-1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last)
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
        logits = self.predictor.get_logits(y_embs, states, attended_states, multi_step=True)
        return logits

class Predictor(object):
    def __init__(self, config, hidden_to_logits_W=None):
        self.config = config
        with tf.name_scope("prev_emb_to_hidden"):
            self.prev_emb_to_hidden = FeedForwardLayer(
                                in_size=config.embedding_size,
                                out_size=config.embedding_size,
                                non_linearity=lambda y: y,
                                use_layer_norm=config.use_layer_norm)
        with tf.name_scope("state_to_hidden"):
            self.state_to_hidden = FeedForwardLayer(
                                    in_size=config.state_size,
                                    out_size=config.embedding_size,
                                    non_linearity=lambda y: y,
                                    use_layer_norm=config.use_layer_norm)
        with tf.name_scope("attended_context_to_hidden"):
            self.att_ctx_to_hidden = FeedForwardLayer(
                                    in_size=2*config.state_size,
                                    out_size=config.embedding_size,
                                    non_linearity=lambda y: y,
                                    use_layer_norm=config.use_layer_norm)

        if config.output_hidden_activation == 'prelu':
            with tf.name_scope("hidden_prelu"):
                self.hidden_prelu = PReLU(in_size=config.embedding_size)

        with tf.name_scope("hidden_to_logits"):
            self.hidden_to_logits = FeedForwardLayer(
                            in_size=config.embedding_size,
                            out_size=config.target_vocab_size,
                            non_linearity=lambda y: y,
                            W=hidden_to_logits_W)

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
            assert(False, 'Unknown output activation function "%s"' % self.config.output_hidden_activation)

        with tf.name_scope("hidden_to_logits"):
            logits = self.hidden_to_logits.forward(hidden, input_is_3d=multi_step)
        
        return logits 


class Encoder(object):
    def __init__(self, config):
        with tf.name_scope("embedding"):
            self.emb_layer = EmbeddingLayer(
                                config.source_vocab_size,
                                config.embedding_size)

        with tf.name_scope("forwardEncoder"):
            self.gru_forward = GRUStep(
                                input_size=config.embedding_size,
                                state_size=config.state_size,
                                use_layer_norm=config.use_layer_norm,
                                nematus_compat=False)

        with tf.name_scope("backwardEncoder"):
            self.gru_backward = GRUStep(
                                    input_size=config.embedding_size,
                                    state_size=config.state_size,
                                    use_layer_norm=config.use_layer_norm,
                                    nematus_compat=False)
        self.state_size = config.state_size

    def get_context(self, x, x_mask):
        with tf.name_scope("embedding"):
            embs = self.emb_layer.forward(x)
            embs_reversed = tf.reverse(embs, axis=[0], name='reverse_embeddings')

        batch_size = tf.shape(x)[1]
        init_state = tf.zeros(shape=[batch_size, self.state_size], dtype=tf.float32)
        with tf.name_scope("forwardEncoder"):
            gates_x, proposal_x = self.gru_forward.precompute_from_x(embs)
            def step_fn(prev_state, x):
                gates_x2d, proposal_x2d = x
                return self.gru_forward.forward(
                        prev_state,
                        gates_x=gates_x2d,
                        proposal_x=proposal_x2d)
            states = RecurrentLayer(
                        initial_state=init_state,
                        step_fn = step_fn).forward((gates_x, proposal_x))

        with tf.name_scope("backwardEncoder"):
            gates_x, proposal_x = self.gru_backward.precompute_from_x(embs_reversed)
            def step_fn(prev_state, x):
                gates_x2d, proposal_x2d, mask = x
                new_state = self.gru_backward.forward(
                                prev_state,
                                gates_x=gates_x2d,
                                proposal_x=proposal_x2d)
                new_state *= mask # batch x 1
                # first couple of states of reversed encoder should be zero
                # this is why we need to multiply by mask
                # this way, when the reversed encoder reaches actual words
                # the state will be zeros and not some accumulated garbage
                return new_state
                
            x_mask_r = tf.reverse(x_mask, axis=[0])
            x_mask_r = tf.expand_dims(x_mask_r, axis=[2]) #seqLen x batch x 1
            states_reversed = RecurrentLayer(
                                initial_state=init_state,
                                step_fn = step_fn).forward((gates_x, proposal_x, x_mask_r))
            states_reversed = tf.reverse(states_reversed, axis=[0])

        concat_states = tf.concat([states, states_reversed], axis=2)
        return concat_states
        
class StandardModel(object):
    def __init__(self, config):

        #variable dimensions
        seqLen = None
        batch_size = None

        self.x = tf.placeholder(
                    dtype=tf.int32,
                    name='x',
                    shape=(seqLen, batch_size))
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

        with tf.name_scope("encoder"):
            self.encoder = Encoder(config)
            ctx = self.encoder.get_context(self.x, self.x_mask)
        
        with tf.name_scope("decoder"):
            self.decoder = Decoder(config, ctx, self.x_mask)
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
        return self.x, self.x_mask, self.y, self.y_mask
    
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
        # x_in, x_mask_in are numpy arrays with shape (seqLen, batch)
        # change init_state, context, context_in_attention_layer
        x_in = numpy.repeat(x_in, repeats=beam_size, axis=1)
        x_mask_in = numpy.repeat(x_mask_in, repeats=beam_size, axis=1)
        feeds = {self.x : x_in, self.x_mask : x_mask_in}
        beam_ys, parents, cost = self._get_beam_search_outputs(beam_size)
        beam_ys_out, parents_out, cost_out = session.run(
                                                    [beam_ys, parents, cost],
                                                    feed_dict=feeds)
        return inference.reconstruct_hypotheses(beam_ys_out, parents_out, cost_out, beam_size)
