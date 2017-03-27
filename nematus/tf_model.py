import tensorflow as tf
from tf_layers import *
import numpy

class Decoder(object):
    def __init__(self, config, context, x_mask):
        with tf.name_scope("next_word_predictor"):
            self.predictor = Predictor(config)

        with tf.name_scope("initial_state_constructor"):
            context_sum = tf.reduce_sum(
                            context * tf.expand_dims(x_mask, axis=2),
                            axis=0)

            context_mean = context_sum / tf.expand_dims(
                                            tf.reduce_sum(x_mask, axis=0),
                                            axis=1)
            self.init_state_layer = FeedForwardLayer(
                                        in_size=config.state_size * 2,
                                        out_size=config.state_size)
            self.init_state = self.init_state_layer.forward(context_mean)

            self.translation_maxlen = config.translation_maxlen
            self.embedding_size = config.embedding_size
            self.state_size = config.state_size
            self.target_vocab_size = config.target_vocab_size

        with tf.name_scope("y_embeddings_layer"):
            self.y_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.target_vocab_size,
                                embedding_size=config.embedding_size)

        if config.use_layer_norm:
            GRUctor = GRUStepWithNormalization
        else:
            GRUctor = GRUStep
        self.grustep1 = GRUctor(
                            input_size=config.embedding_size,
                            state_size=config.state_size)
        self.attstep = AttentionStep(
                        context=context,
                        context_state_size=2*config.state_size,
                        context_mask=x_mask,
                        state_size=config.state_size,
                        hidden_size=2*config.state_size)
        self.grustep2 = GRUctor(
                            input_size=2*config.state_size,
                            state_size=config.state_size,
                            nematus_compat=config.nematus_compat)

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

    def beam_search(self, beam_size):

        """
        Strategy:
            compute the log_probs - same as with sampling
            for sentences that are ended set log_prob(<eos>)=0, log_prob(not eos)=-inf
            add previous cost to log_probs
            run top k -> (idxs, values)
            use values as new costs
            divide idxs by num_classes to get state_idxs
            use gather to get new states
            take the remainder of idxs after num_classes to get new_predicted words
        """

        # Initialize loop variables
        batch_size = tf.shape(self.init_state)[0]
        i = tf.constant(0)
        init_ys = -tf.ones(dtype=tf.int32, shape=[batch_size])
        init_embs = tf.zeros(dtype=tf.float32, shape=[batch_size,self.embedding_size])

        f_min = numpy.finfo(numpy.float32).min
        init_cost = [0.] + [f_min]*(beam_size-1) # to force first top k are from first hypo only
        init_cost = tf.constant(init_cost, dtype=tf.float32)
        init_cost = tf.tile(init_cost, multiples=[batch_size/beam_size])
        ys_array = tf.TensorArray(
                    dtype=tf.int32,
                    size=self.translation_maxlen,
                    clear_after_read=True,
                    name='y_sampled_array')
        p_array = tf.TensorArray(
                    dtype=tf.int32,
                    size=self.translation_maxlen,
                    clear_after_read=True,
                    name='parent_idx_array')
        init_loop_vars = [i, self.init_state, init_ys, init_embs, init_cost, ys_array, p_array]

        # Prepare cost matrix for completed sentences -> Prob(EOS) = 1 and Prob(x) = 0
        eos_log_probs = tf.constant(
                            [[0.] + ([f_min]*(self.target_vocab_size - 1))],
                            dtype=tf.float32)
        eos_log_probs = tf.tile(eos_log_probs, multiples=[batch_size,1])

        def cond(i, states, prev_ys, prev_embs, cost, ys_array, p_array):
            return tf.logical_and(
                    tf.less(i, self.translation_maxlen),
                    tf.reduce_any(tf.not_equal(prev_ys, 0)))

        def body(i, states, prev_ys, prev_embs, cost, ys_array, p_array):
            #If ensemble decoding is necessary replace with for loop and do model[i].{grustep1,attstep,...}
            new_states1 = self.grustep1.forward(states, prev_embs)
            att_ctx = self.attstep.forward(new_states1)
            new_states2 = self.grustep2.forward(new_states1, att_ctx)
            logits = self.predictor.get_logits(prev_embs, new_states2, att_ctx, multi_step=False)
            log_probs = tf.nn.log_softmax(logits) # shape (batch, vocab_size)

            # set cost of EOS to zero for completed sentences so that they are in top k
            # Need to make sure only EOS is selected because a completed sentence might
            # kill ongoing sentences
            log_probs = tf.where(tf.equal(prev_ys, 0), eos_log_probs, log_probs)

            all_costs = log_probs + tf.expand_dims(cost, axis=1) # TODO: you might be getting NaNs here since -inf is in log_probs

            all_costs = tf.reshape(all_costs, shape=[-1, self.target_vocab_size * beam_size])
            values, indices = tf.nn.top_k(all_costs, k=beam_size) #the sorted option is by default True, is this needed? 
            new_cost = tf.reshape(values, shape=[batch_size])
            offsets = tf.range(
                        start = 0,
                        delta = beam_size,
                        limit = batch_size,
                        dtype=tf.int32)
            offsets = tf.expand_dims(offsets, axis=1)
            survivor_idxs = (indices/self.target_vocab_size) + offsets
            new_ys = indices % self.target_vocab_size
            survivor_idxs = tf.reshape(survivor_idxs, shape=[batch_size])
            new_ys = tf.reshape(new_ys, shape=[batch_size])
            new_embs = self.y_emb_layer.forward(new_ys)
            new_states = tf.gather(new_states2, indices=survivor_idxs)

            new_cost = tf.where(tf.equal(new_ys, 0), tf.abs(new_cost), new_cost)

            ys_array = ys_array.write(i, value=new_ys)
            p_array = p_array.write(i, value=survivor_idxs)

            return i+1, new_states, new_ys, new_embs, new_cost, ys_array, p_array


        final_loop_vars = tf.while_loop(
                            cond=cond,
                            body=body,
                            loop_vars=init_loop_vars,
                            back_prop=False)
        i, _, _, _, cost, ys_array, p_array = final_loop_vars

        indices = tf.range(0, i)
        sampled_ys = ys_array.gather(indices)
        parents = p_array.gather(indices)
        cost = tf.abs(cost) #to get negative-log-likelihood
        return sampled_ys, parents, cost

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
    def __init__(self, config):
        with tf.name_scope("prev_emb_to_hidden"):
            self.prev_emb_to_hidden = FeedForwardLayer(
                                in_size=config.embedding_size,
                                out_size=config.embedding_size,
                                non_linearity=lambda y: y)
        with tf.name_scope("state_to_hidden"):
            self.state_to_hidden = FeedForwardLayer(
                                    in_size=config.state_size,
                                    out_size=config.embedding_size,
                                    non_linearity=lambda y: y)
        with tf.name_scope("attended_context_to_hidden"):
            self.att_ctx_to_hidden = FeedForwardLayer(
                                    in_size=2*config.state_size,
                                    out_size=config.embedding_size,
                                    non_linearity=lambda y: y)
        with tf.name_scope("hidden_to_logits"):
            self.hidden_to_logits = FeedForwardLayer(
                            in_size=config.embedding_size,
                            out_size=config.target_vocab_size,
                            non_linearity=lambda y: y)

    def get_logits(self, y_embs, states, attended_states, multi_step=True):
        with tf.name_scope("prev_emb_to_hidden"):
            hidden_emb = self.prev_emb_to_hidden.forward(y_embs, input_is_3d=multi_step)

        with tf.name_scope("state_to_hidden"):
            hidden_state = self.state_to_hidden.forward(states, input_is_3d=multi_step)

        with tf.name_scope("attended_context_to_hidden"):
            hidden_att_ctx = self.att_ctx_to_hidden.forward(attended_states,input_is_3d=multi_step)

        hidden = hidden_emb + hidden_state + hidden_att_ctx
        hidden = tf.tanh(hidden)

        with tf.name_scope("hidden_to_logits"):
            logits = self.hidden_to_logits.forward(hidden, input_is_3d=multi_step)
        
        return logits 


class Encoder(object):
    def __init__(self, config):
        with tf.name_scope("embedding"):
            self.emb_layer = EmbeddingLayer(
                                config.source_vocab_size,
                                config.embedding_size)

        if config.use_layer_norm:
            GRUctor = GRUStepWithNormalization
        else:
            GRUctor = GRUStep
        with tf.name_scope("forwardEncoder"):
            self.gru_forward = GRUctor(
                                input_size=config.embedding_size,
                                state_size=config.state_size)

        with tf.name_scope("backwardEncoder"):
            self.gru_backward = GRUctor(
                                    input_size=config.embedding_size,
                                    state_size=config.state_size)
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
        batch_size = config.batch_size

        if config.translate_valid:
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

        #with tf.name_scope("optimizer"):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
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

    def get_global_step(self):
        return self.t

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
            self.beam_ys, self.parents, self.cost =  self.decoder.beam_search(beam_size)
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
        hypotheses = self._reconstruct(beam_ys_out, parents_out, cost_out, beam_size)
        return hypotheses

    def _reconstruct(self, ys, parents, cost, beam_size):
        #ys.shape = parents.shape = (seqLen, beam_size x batch_size) 
        # output: hypothesis list with shape (batch_size, beam_size, (sequence, cost))

        def reconstruct_single(ys, parents, hypoId, hypo, pos):
            if pos < 0:
                hypo.reverse()
                return hypo
            else:
                hypo.append(ys[pos, hypoId])
                hypoId = parents[pos, hypoId]
                return reconstruct_single(ys, parents, hypoId, hypo, pos - 1)

        hypotheses = []
        batch_size = ys.shape[1] / beam_size
        pos = ys.shape[0] - 1
        for batch in range(batch_size):
            hypotheses.append([])
            for beam in range(beam_size):
                i = batch*beam_size + beam
                hypo = reconstruct_single(ys, parents, i, [], pos)
                hypo = numpy.trim_zeros(hypo, trim='b') # b for back
                hypo.append(0)
                hypotheses[batch].append((hypo, cost[i]))
        return hypotheses
