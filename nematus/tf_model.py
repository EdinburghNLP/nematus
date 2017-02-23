import tensorflow as tf
from tf_layers import *

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
            init_attended_context = tf.zeros([tf.shape(self.init_state)[0], config.state_size*2])
            self.init_state = (self.init_state, init_attended_context)

        with tf.name_scope("y_embeddings_layer"):
            self.y_emb_layer = EmbeddingLayer(
                                vocabulary_size=config.target_vocab_size,
                                embedding_size=config.embedding_size)

        self.grustep1 = GRUStep(
                            input_size=config.embedding_size,
                            state_size=config.state_size)
        self.attstep = AttentionStep(
                        context=context,
                        context_state_size=2*config.state_size,
                        context_mask=x_mask,
                        state_size=config.state_size,
                        hidden_size=2*config.state_size)
        self.grustep2 = GRUStep(
                            input_size=2*config.state_size,
                            state_size=config.state_size)

    def sample(self):
       ##use self.init_state
       #def cond(loop_vars):
       #    i, states, prev_ys, prev_embs = loop_vars
       #    return tf.logical_and(
       #            tf.less(i, maxlen),
       #            tf.reduce_all(tf.equal(prev_ys, 0))) #tf.zeros(shape=tf.shape(prev_ys), dtype=tf.int32))))

       #def body(loop_vars):
       #    i, states, prev_ys, prev_embs = loop_vars
       #    new_states1 = gru_layer1.forward(states, prev_embs)
       #    att_ctx = att_layer.forward(new_states1)
       #    new_states2 = gru_layer2.forward(new_state1, att_ctx)
       #    logits = prediction_layer.forward(states, prev_embs, att_ctx)
       #    new_ys = tf.multinomial(logits, num_samples=1)
       #    new_ys = tf.squeeze(new_ys, axis=1)
       #    new_ys = tf.where(tf.equal(prev_ys, 0), 0, new_ys)
       #    new_embs = embs_layer.forward(new_ys)
       #    return i+1, new_states2, new_ys, new_embs

       #_, _, ys, _ = tf.while_loop(
       #                    cond=cond,
       #                    body=body,
       #                    loop_vars=loop_vars
       #                    back_prop=False)
       #sampled_ys = ys.stack() # seqLen X batch -- has trailing zeros
       #return sampled_ys
       pass

    def beam_search(self, context, x_mask, maxlen, beam_size):
        """
        Strategy:
            tile context and init_state
            compute the log_probs
            add previous cost to log_probs
            flatten cost
            set cost of class 0 to 0 for each beam that has already ended
            e.g. tf.where(mask == True, 0, cost)
            run top k -> (idxs, values)
            use values as new costs
            divide idxs by num_classes to get state_idxs
            use gather to get new states
            take the remainder of idxs after num_classes to get new_predicted words
            use gather to get new mask
            update the mask (finished?) according to new_predicted_words, e.g. tf.logical_or(mask, tf.equal(new_predicted_words, 0))
        Now try to do in batches:
        mask.shape (batch, beam)
        log_probs.shape(batch, beam, num_classes)
        context.shape (seqLen, batch, emb_size)
        """
          
        return None #beam_ys

    def score(self, y):
        with tf.name_scope("y_embeddings_layer"):
            y_but_last = tf.slice(y, [0,0], [tf.shape(y)[0]-1, -1])
            y_embs = self.y_emb_layer.forward(y_but_last)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        def step_fn(prev, x):
            prev_state = prev[0]
            prev_att_ctx = prev[1]
            state = self.grustep1.forward(prev_state, x)
            att_ctx = self.attstep.forward(state) 
            state = self.grustep2.forward(state, att_ctx)
            #TODO: write att_ctx to tensorArray instead of having it as output of scan?
            return (state, att_ctx)

        states, attended_states = RecurrentLayer(
                                    initial_state=self.init_state,
                                    step_fn=step_fn).forward(y_embs)
        logits = self.predictor.get_logits(y_embs, states, attended_states, multi_step=True)
        return logits

class Predictor(object):
    def __init__(self, config):
        with tf.name_scope("prev_emb_to_hidden"):
            self.prev_emb_to_hidden = FeedForwardLayer(
                                in_size=config.embedding_size,
                                out_size=config.embedding_size)
        with tf.name_scope("state_to_hidden"):
            self.state_to_hidden = FeedForwardLayer(
                                    in_size=config.state_size,
                                    out_size=config.embedding_size)
        with tf.name_scope("attended_context_to_hidden"):
            self.att_ctx_to_hidden = FeedForwardLayer(
                                    in_size=2*config.state_size,
                                    out_size=config.embedding_size)
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
            self.gru1 = GRUStep(
                            input_size=config.embedding_size,
                            state_size=config.state_size)

        with tf.name_scope("backwardEncoder"):
            self.gru2 = GRUStep(
                    input_size=config.embedding_size,
                    state_size=config.state_size)
        self.state_size = config.state_size

    def get_context(self, x):
        with tf.name_scope("embedding"):
            embs = self.emb_layer.forward(x)
            embs_reversed = tf.reverse(embs, axis=[0], name='reverse_embeddings')

        batch_size = tf.shape(x)[1]
        init_state = tf.zeros(shape=[batch_size, self.state_size], dtype=tf.float32)
        with tf.name_scope("forwardEncoder"):
            def step_fn(prev_state, x):
                return self.gru1.forward(prev_state, x)
            states = RecurrentLayer(
                        initial_state=init_state,
                        step_fn = step_fn).forward(embs)

        with tf.name_scope("backwardEncoder"):
            def step_fn(prev_state, x):
                return self.gru2.forward(prev_state, x)
            states_reversed = RecurrentLayer(
                                initial_state=init_state,
                                step_fn = step_fn).forward(embs_reversed)
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
            ctx = self.encoder.get_context(self.x)
        
        with tf.name_scope("decoder"):
            self.decoder = Decoder(config, ctx, self.x_mask)
            self.logits = self.decoder.score(self.y)

        with tf.name_scope("loss"):
            self.loss_layer = Masked_cross_entropy_loss(self.y, self.y_mask)
            self.loss_per_sentence = self.loss_layer.forward(self.logits)

    def get_score_inputs(self):
        return self.x, self.x_mask, self.y, self.y_mask
    
    def get_loss(self):
        return self.loss_per_sentence

