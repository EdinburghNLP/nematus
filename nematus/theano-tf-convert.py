#!/usr/bin/env python 
import numpy as np
import tensorflow as tf
from tf_nmt import create_model
import argparse

th2tf = {
    'Wemb' : 'encoder/embedding/embeddings:0',
    'Wemb_dec' : 'decoder/y_embeddings_layer/embeddings:0',
    'decoder_U' : 'decoder/state_to_gates:0',
    'decoder_U_att' : 'decoder/hidden_to_score:0',
    'decoder_U_nl' : 'decoder/state_to_gates_1:0',
    'decoder_Ux' : 'decoder/state_to_proposal:0',
    'decoder_Ux_nl' : 'decoder/state_to_proposal_1:0',
    'decoder_W' : 'decoder/input_to_gates:0',
    'decoder_W_comb_att' : 'decoder/state_to_hidden:0',
    'decoder_Wc' : 'decoder/input_to_gates_1:0',
    'decoder_Wc_att' : 'decoder/context_to_hidden:0',
    'decoder_Wcx' : 'decoder/input_to_proposal_1:0',
    'decoder_Wx' : 'decoder/input_to_proposal:0',
    'decoder_b' : 'decoder/gates_bias:0',
    'decoder_b_att' : 'decoder/hidden_bias:0',
    'decoder_b_nl' : 'decoder/gates_bias_1:0',
    'decoder_bx' : 'decoder/proposal_bias:0',
    'decoder_bx_nl' : 'decoder/proposal_bias_1:0',
    'decoder_c_tt' : None,
    'encoder_U' : 'encoder/forwardEncoder/state_to_gates:0',
    'encoder_Ux' : 'encoder/forwardEncoder/state_to_proposal:0',
    'encoder_W' : 'encoder/forwardEncoder/input_to_gates:0',
    'encoder_Wx' : 'encoder/forwardEncoder/input_to_proposal:0',
    'encoder_b' : 'encoder/forwardEncoder/gates_bias:0',
    'encoder_bx' : 'encoder/forwardEncoder/proposal_bias:0',
    'encoder_r_U' : 'encoder/backwardEncoder/state_to_gates:0',
    'encoder_r_Ux' : 'encoder/backwardEncoder/state_to_proposal:0',
    'encoder_r_W' : 'encoder/backwardEncoder/input_to_gates:0',
    'encoder_r_Wx' : 'encoder/backwardEncoder/input_to_proposal:0',
    'encoder_r_b' : 'encoder/backwardEncoder/gates_bias:0',
    'encoder_r_bx' : 'encoder/backwardEncoder/proposal_bias:0',
    'ff_logit_W' : 'decoder/next_word_predictor/hidden_to_logits/W:0',
    'ff_logit_b' : 'decoder/next_word_predictor/hidden_to_logits/b:0',
    'ff_logit_ctx_W' : 'decoder/next_word_predictor/attended_context_to_hidden/W:0',
    'ff_logit_ctx_b' : 'decoder/next_word_predictor/attended_context_to_hidden/b:0',
    'ff_logit_lstm_W' : 'decoder/next_word_predictor/state_to_hidden/W:0',
    'ff_logit_lstm_b' : 'decoder/next_word_predictor/state_to_hidden/b:0',
    'ff_logit_prev_W' : 'decoder/next_word_predictor/prev_emb_to_hidden/W:0',
    'ff_logit_prev_b' : 'decoder/next_word_predictor/prev_emb_to_hidden/b:0',
    'ff_state_W' : 'decoder/initial_state_constructor/W:0',
    'ff_state_b' : 'decoder/initial_state_constructor/b:0',
    'history_errs' : None,
    'uidx' : 'time:0'}

class FakeConfig(object):
    def __init__(self, state_size, embedding_size, source_vocab_size, target_vocab_size):
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        # These are needed to create the model, but their values are irrelevant
        self.reload = False
        self.maxlen = None
        self.learning_rate = 0.0001
        self.clip_c = 1


def theano_to_tensorflow_model(in_path, out_path):
    saved_model = np.load(in_path)

    # Create fake config
    src_size, src_emb_size = saved_model['Wemb'].shape
    trg_size, trg_emb_size = saved_model['Wemb_dec'].shape
    state_size = saved_model['decoder_U'].shape[0]
    assert trg_emb_size == src_emb_size, 'src_emb_size ({}) and trg_emb_size ({}) do not equal, this is unsupported in TF'.format(src_emb_size, trg_emb_size)
    fake_config = FakeConfig(state_size, src_emb_size, src_size, trg_size)

    with tf.Session() as sess:
        model, saver = create_model(fake_config, sess)
        seen = set()
        assign_ops = []
        for key in saved_model.keys():
            tf_name = th2tf[key]
            if tf_name is not None:
                assert tf_name not in seen
                seen.add(tf_name)
                tf_var = tf.get_default_graph().get_tensor_by_name(tf_name)
                assign_ops.append(tf.assign(tf_var, saved_model[key]))
            else:
                print "Not saving", key, "because no TF equivalent"
        sess.run(assign_ops)
        saver.save(sess, save_path=out_path)


def tensorflow_to_theano_model(in_path, out_path):
    keys, values = zip(*th2tf.items())
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(in_path + '.meta')
        new_saver.restore(sess, in_path)
        params = {}
        for th_name, tf_name in th2tf.items():
            if tf_name is not None:
                v = sess.run(tf.get_default_graph().get_tensor_by_name(tf_name))
            else:
                if th_name == 'history_errs':
                    v = []
                elif th_name == 'decoder_c_tt':
                    v = np.zeros(1, dtype=np.float32)
                else:
                    assert False, 'Need to handle {}'.format(th_name)
            assert th_name not in params, '{} is repeated!'.format(th_name)
            params[th_name] = v
    np.savez(out_path, **params)
    print 'Saved {} params in {}'.format(len(params), out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from_theano', action='store_true')
    group.add_argument('--from_tf', action='store_true')

    data = parser.add_argument_group()
    data.add_argument('--in', type=str, required=True, metavar='PATH', dest='inn')
    data.add_argument('--out', type=str, required=True, metavar='PATH')

    opts = parser.parse_args()
    
    if opts.from_theano:
        theano_to_tensorflow_model(opts.inn, opts.out)
    elif opts.from_tf:
        tensorflow_to_theano_model(opts.inn, opts.out)
    else:
        assert False


