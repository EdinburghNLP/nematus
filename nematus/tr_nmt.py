import tensorflow as tf
from tf_layers import *
from collections import namedtuple

Config = namedtuple('Config', [
    'source_vocab_size',
    'target_vocab_size',
    'embedding_size',
    'state_size'])

def build_model(config):

    x, x_mask, context = build_encoder(config)
    y, y_mask, y_embs, states, attended_states = build_decoder(config, context, x_mask)
    logits = build_predictor(config, y_embs, states, attended_states)
    
    return x, x_mask, y, y_mask, logits

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
        logits = FeedForwardLayer(
                    in_size=config.embedding_size,
                    out_size=config.target_vocab_size,
                    non_linearity=lambda y: y)
    return logits 



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
            




    


