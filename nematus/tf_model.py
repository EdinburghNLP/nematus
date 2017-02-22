import tensorflow as tf
from tf_layers import *

def build_model(config):

    x, x_mask, context = build_encoder(config)
    y, y_mask, y_embs, states, attended_states = build_decoder(config, context, x_mask)
    logits = build_predictor(config, y_embs, states, attended_states)
    
    return x, x_mask, y, y_mask, logits

def build_encoder(config):
    with tf.name_scope("encoder"):
        x = tf.placeholder(dtype=tf.int32,
                        name='x',
                        shape=(None, None)) #seqLen X batch
        x_mask = tf.placeholder(dtype=tf.float32,
                                name='x_mask',
                                shape=(None, None))
        batch_size = tf.shape(x)[1]
        with tf.name_scope("embedding"):
            embs = EmbeddingLayer(config.source_vocab_size,
                                  config.embedding_size).forward(x)
            embs_reversed = tf.reverse(embs, axis=[0], name='reverse_embeddings')

        with tf.name_scope("forwardEncoder"):
            init_state = tf.zeros(shape=[batch_size, config.state_size], dtype=tf.float32)
            gru1 = GRUStep(
                    input_size=config.embedding_size,
                    state_size=config.state_size)
            step_fn = lambda prev_state, x: gru1.forward(prev_state, x)
            states = RecurrentLayer(
                        initial_state=init_state,
                        step_fn = step_fn).forward(embs)

        with tf.name_scope("backwardEncoder"):
            init_state = tf.zeros(shape=[batch_size, config.state_size], dtype=tf.float32)
            gru2 = GRUStep(
                    input_size=config.embedding_size,
                    state_size=config.state_size)
            step_fn = lambda prev_state, x: gru2.forward(prev_state, x)
            states_reversed = RecurrentLayer(
                                initial_state=init_state,
                                step_fn = step_fn).forward(embs_reversed)
            states_reversed = tf.reverse(states_reversed, axis=[0])

        concat_states = tf.concat([states, states_reversed], axis=2)
        return x, x_mask, concat_states
            
def build_decoder(config, context, x_mask):
    with tf.name_scope("decoder"):
        y = tf.placeholder(dtype=tf.int32,
                           name='y',
                           shape=(None, None)) # seqLen X batch
        y_mask = tf.placeholder(dtype=tf.float32,
                                name='y_mask',
                                shape=(None, None))

        # x_mask shape: seqLen,batch
        # context shape: seqLen,batch,state_size
        context_sum = tf.reduce_sum(
                        context * tf.expand_dims(x_mask, axis=2),
                        axis=0)
        context_mean = context_sum / tf.expand_dims(
                                        tf.reduce_sum(x_mask, axis=0),
                                        axis=1)

        with tf.name_scope("initial_state_constructor"):
            init_state = FeedForwardLayer(
                            in_size=config.state_size * 2,
                            out_size=config.state_size).forward(context_mean)
            init_attended_context = tf.zeros([tf.shape(init_state)[0], config.state_size*2])
            init_state = (init_state, init_attended_context)

        with tf.name_scope("y_embeddings_layer"):
            y_but_last = tf.slice(y, [0,0], [tf.shape(y)[0]-1, -1])
            y_embs = EmbeddingLayer(
                        vocabulary_size=config.target_vocab_size,
                        embedding_size=config.embedding_size).forward(y_but_last)
            y_embs = tf.pad(y_embs,
                            mode='CONSTANT',
                            paddings=[[1,0],[0,0],[0,0]]) # prepend zeros

        grustep1 = GRUStep(
                    input_size=config.embedding_size,
                    state_size=config.state_size)
        attstep = AttentionStep(
                    context=context,
                    context_state_size=2*config.state_size,
                    context_mask=x_mask,
                    state_size=config.state_size,
                    hidden_size=2*config.state_size)
        grustep2 = GRUStep(
                    input_size=2*config.state_size,
                    state_size=config.state_size)
        def step_fn(prev, x):
            prev_state = prev[0]
            prev_att_ctx = prev[1]
            state = grustep1.forward(prev_state, x)
            att_ctx = attstep.forward(state) 
            state = grustep2.forward(state, att_ctx)
            return (state, att_ctx)

        states, attended_states = RecurrentLayer(
                                    initial_state=init_state,
                                    step_fn=step_fn).forward(y_embs)

        return y, y_mask, y_embs, states, attended_states

# def build_sampler(config, context, x_mask):
#     back_prop=False
# 
#     #TODO: self.gru_layer1, self.gru_layer2, self.att_layer, self.prediction_layer
# 
#     def cond(loop_vars):
#         i, states, prev_ys, prev_embs = loop_vars
#         return tf.logical_and(
#                 tf.less(i, maxlen),
#                 tf.reduce_all(tf.equal(prev_ys, 0))) #tf.zeros(shape=tf.shape(prev_ys), dtype=tf.int32))))
# 
#     def body(loop_vars):
#         i, states, prev_ys, prev_embs = loop_vars
#         new_states1 = gru_layer1.forward(states, prev_embs)
#         att_ctx = att_layer.forward(new_states1)
#         new_states2 = gru_layer2.forward(new_state1, att_ctx)
#         logits = prediction_layer.forward(states, prev_embs, att_ctx)
#         new_ys = tf.multinomial(logits, num_samples=1)
#         new_ys = tf.squeeze(new_ys, axis=1)
#         new_ys = tf.where(tf.equal(prev_ys, 0), 0, new_ys)
#         new_embs = embs_layer.forward(new_ys)
#         return i+1, new_states2, new_ys, new_embs
# 
#     _, _, ys, _ = tf.while_loop(
#                         cond=cond,
#                         body=body,
#                         loop_vars=loop_vars
#                         back_prop=False)
#    ys = ys.stack() # seqLen X batch -- has trailing zeros
#    return ys
# 
# def build_beam_searcher(config, context, x_mask):
#     beam_size = tf.constant(config.beam_size, dtype=tf.int32)
#     """
#     Strategy:
#         tile context and init_state
#         compute the log_probs
#         add previous cost to log_probs
#         flatten cost
#         set cost of class 0 to 0 for each beam that has already ended
#         e.g. tf.where(mask == True, 0, cost)
#         run top k -> (idxs, values)
#         use values as new costs
#         divide idxs by num_classes to get state_idxs
#         use gather to get new states
#         take the remainder of idxs after num_classes to get new_predicted words
#         use gather to get new mask
#         update the mask (finished?) according to new_predicted_words, e.g. tf.logical_or(mask, tf.equal(new_predicted_words, 0))
#     Now try to do in batches:
#     mask.shape (batch, beam)
#     log_probs.shape(batch, beam, num_classes)
#     context.shape (seqLen, batch, emb_size)
#     """
#     
        



                           
def build_predictor(config, y_embs, states, attended_states):
    with tf.name_scope("next_word_predictor"):
        with tf.name_scope("prev_emb_to_hidden"):
            hidden_emb = FeedForwardLayer(
                            in_size=config.embedding_size,
                            out_size=config.embedding_size,
                            input_is_3d=True).forward(y_embs)
        with tf.name_scope("state_to_hidden"):
            hidden_state = FeedForwardLayer(
                            in_size=config.state_size,
                            out_size=config.embedding_size,
                            input_is_3d=True).forward(states)
        with tf.name_scope("attended_context_to_hidden"):
            hidden_att_ctx = FeedForwardLayer(
                                in_size=2*config.state_size,
                                out_size=config.embedding_size,
                                input_is_3d=True).forward(attended_states)
        hidden = hidden_emb + hidden_state + hidden_att_ctx
        with tf.name_scope("hidden_to_logits"):
            logits = FeedForwardLayer(
                        in_size=config.embedding_size,
                        out_size=config.target_vocab_size,
                        non_linearity=lambda y: y,
                        input_is_3d=True).forward(hidden)
    return logits 


