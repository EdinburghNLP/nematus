#!/usr/bin/env python 

import argparse
import os

import numpy as np
import tensorflow as tf

import compat
import nmt
import util

def construct_parameter_map(config):
    th2tf = {
        # encoder/embedding
        'Wemb' : 'encoder/embedding/embeddings:0',

        # encoder/forward-stack
        'encoder_U' : 'encoder/forward-stack/level0/gru0/state_to_gates:0',
        'encoder_U_lnb' : 'encoder/forward-stack/level0/gru0/gates_state_norm/new_mean:0',
        'encoder_U_lns' : 'encoder/forward-stack/level0/gru0/gates_state_norm/new_std:0',
        'encoder_Ux' : 'encoder/forward-stack/level0/gru0/state_to_proposal:0',
        'encoder_Ux_lnb' : 'encoder/forward-stack/level0/gru0/proposal_state_norm/new_mean:0',
        'encoder_Ux_lns' : 'encoder/forward-stack/level0/gru0/proposal_state_norm/new_std:0',
        'encoder_W' : 'encoder/forward-stack/level0/gru0/input_to_gates:0',
        'encoder_W_lnb' : 'encoder/forward-stack/level0/gru0/gates_x_norm/new_mean:0',
        'encoder_W_lns' : 'encoder/forward-stack/level0/gru0/gates_x_norm/new_std:0',
        'encoder_Wx' : 'encoder/forward-stack/level0/gru0/input_to_proposal:0',
        'encoder_Wx_lnb' : 'encoder/forward-stack/level0/gru0/proposal_x_norm/new_mean:0',
        'encoder_Wx_lns' : 'encoder/forward-stack/level0/gru0/proposal_x_norm/new_std:0',
        'encoder_b' : 'encoder/forward-stack/level0/gru0/gates_bias:0',
        'encoder_bx' : 'encoder/forward-stack/level0/gru0/proposal_bias:0',

        # encoder/backward-stack
        'encoder_r_U' : 'encoder/backward-stack/level0/gru0/state_to_gates:0',
        'encoder_r_U_lnb' : 'encoder/backward-stack/level0/gru0/gates_state_norm/new_mean:0',
        'encoder_r_U_lns' : 'encoder/backward-stack/level0/gru0/gates_state_norm/new_std:0',
        'encoder_r_Ux' : 'encoder/backward-stack/level0/gru0/state_to_proposal:0',
        'encoder_r_Ux_lnb' : 'encoder/backward-stack/level0/gru0/proposal_state_norm/new_mean:0',
        'encoder_r_Ux_lns' : 'encoder/backward-stack/level0/gru0/proposal_state_norm/new_std:0',
        'encoder_r_W' : 'encoder/backward-stack/level0/gru0/input_to_gates:0',
        'encoder_r_W_lnb' : 'encoder/backward-stack/level0/gru0/gates_x_norm/new_mean:0',
        'encoder_r_W_lns' : 'encoder/backward-stack/level0/gru0/gates_x_norm/new_std:0',
        'encoder_r_Wx' : 'encoder/backward-stack/level0/gru0/input_to_proposal:0',
        'encoder_r_Wx_lnb' : 'encoder/backward-stack/level0/gru0/proposal_x_norm/new_mean:0',
        'encoder_r_Wx_lns' : 'encoder/backward-stack/level0/gru0/proposal_x_norm/new_std:0',
        'encoder_r_b' : 'encoder/backward-stack/level0/gru0/gates_bias:0',
        'encoder_r_bx' : 'encoder/backward-stack/level0/gru0/proposal_bias:0',

        # decoder/initial_state_constructor
        'ff_state_W' : 'decoder/initial_state_constructor/W:0',
        'ff_state_b' : 'decoder/initial_state_constructor/b:0',
        'ff_state_ln_b' : 'decoder/initial_state_constructor/new_mean:0',
        'ff_state_ln_s' : 'decoder/initial_state_constructor/new_std:0',

        # decoder/y_embeddings_layer
        'Wemb_dec' : 'decoder/y_embeddings_layer/embeddings:0',

        # decoder/base/gru0
        'decoder_U' : 'decoder/base/gru0/state_to_gates:0',
        'decoder_U_lnb' : 'decoder/base/gru0/gates_state_norm/new_mean:0',
        'decoder_U_lns' : 'decoder/base/gru0/gates_state_norm/new_std:0',
        'decoder_Ux' : 'decoder/base/gru0/state_to_proposal:0',
        'decoder_Ux_lnb' : 'decoder/base/gru0/proposal_state_norm/new_mean:0',
        'decoder_Ux_lns' : 'decoder/base/gru0/proposal_state_norm/new_std:0',
        'decoder_W' : 'decoder/base/gru0/input_to_gates:0',
        'decoder_W_lnb' : 'decoder/base/gru0/gates_x_norm/new_mean:0',
        'decoder_W_lns' : 'decoder/base/gru0/gates_x_norm/new_std:0',
        'decoder_Wx' : 'decoder/base/gru0/input_to_proposal:0',
        'decoder_Wx_lnb' : 'decoder/base/gru0/proposal_x_norm/new_mean:0',
        'decoder_Wx_lns' : 'decoder/base/gru0/proposal_x_norm/new_std:0',
        'decoder_b' : 'decoder/base/gru0/gates_bias:0',
        'decoder_bx' : 'decoder/base/gru0/proposal_bias:0',

        # decoder/base/attention
        'decoder_U_att' : 'decoder/base/attention/hidden_to_score:0',
        'decoder_W_comb_att' : 'decoder/base/attention/state_to_hidden:0',
        'decoder_W_comb_att_lnb' : 'decoder/base/attention/hidden_state_norm/new_mean:0',
        'decoder_W_comb_att_lns' : 'decoder/base/attention/hidden_state_norm/new_std:0',
        'decoder_Wc_att' : 'decoder/base/attention/context_to_hidden:0',
        'decoder_Wc_att_lnb' : 'decoder/base/attention/hidden_context_norm/new_mean:0',
        'decoder_Wc_att_lns' : 'decoder/base/attention/hidden_context_norm/new_std:0',
        'decoder_b_att' : 'decoder/base/attention/hidden_bias:0',

        # decoder/base/gru1
        'decoder_U_nl' : 'decoder/base/gru1/state_to_gates:0',
        'decoder_U_nl_lnb' : 'decoder/base/gru1/gates_state_norm/new_mean:0',
        'decoder_U_nl_lns' : 'decoder/base/gru1/gates_state_norm/new_std:0',
        'decoder_Ux_nl' : 'decoder/base/gru1/state_to_proposal:0',
        'decoder_Ux_nl_lnb' : 'decoder/base/gru1/proposal_state_norm/new_mean:0',
        'decoder_Ux_nl_lns' : 'decoder/base/gru1/proposal_state_norm/new_std:0',
        'decoder_Wc' : 'decoder/base/gru1/input_to_gates:0',
        'decoder_Wc_lnb' : 'decoder/base/gru1/gates_x_norm/new_mean:0',
        'decoder_Wc_lns' : 'decoder/base/gru1/gates_x_norm/new_std:0',
        'decoder_Wcx' : 'decoder/base/gru1/input_to_proposal:0',
        'decoder_Wcx_lnb' : 'decoder/base/gru1/proposal_x_norm/new_mean:0',
        'decoder_Wcx_lns' : 'decoder/base/gru1/proposal_x_norm/new_std:0',
        'decoder_b_nl' : 'decoder/base/gru1/gates_bias:0',
        'decoder_bx_nl' : 'decoder/base/gru1/proposal_bias:0',

        # decoder/next_word_predictor
        'ff_logit_W' : 'decoder/next_word_predictor/hidden_to_logits/W:0',
        'ff_logit_b' : 'decoder/next_word_predictor/hidden_to_logits/b:0',
        'ff_logit_ctx_W' : 'decoder/next_word_predictor/attended_context_to_hidden/W:0',
        'ff_logit_ctx_b' : 'decoder/next_word_predictor/attended_context_to_hidden/b:0',
        'ff_logit_ctx_ln_b' : 'decoder/next_word_predictor/attended_context_to_hidden/new_mean:0',
        'ff_logit_ctx_ln_s' : 'decoder/next_word_predictor/attended_context_to_hidden/new_std:0',
        'ff_logit_lstm_W' : 'decoder/next_word_predictor/state_to_hidden/W:0',
        'ff_logit_lstm_b' : 'decoder/next_word_predictor/state_to_hidden/b:0',
        'ff_logit_lstm_ln_b' : 'decoder/next_word_predictor/state_to_hidden/new_mean:0',
        'ff_logit_lstm_ln_s' : 'decoder/next_word_predictor/state_to_hidden/new_std:0',
        'ff_logit_prev_W' : 'decoder/next_word_predictor/prev_emb_to_hidden/W:0',
        'ff_logit_prev_b' : 'decoder/next_word_predictor/prev_emb_to_hidden/b:0',
        'ff_logit_prev_ln_b' : 'decoder/next_word_predictor/prev_emb_to_hidden/new_mean:0',
        'ff_logit_prev_ln_s' : 'decoder/next_word_predictor/prev_emb_to_hidden/new_std:0',

        # other
        'decoder_c_tt' : None,
        'history_errs' : None,
        'uidx' : 'time:0'}

    for i in range(1, len(config.dim_per_factor)):
        th_name = 'Wemb{0}'.format(i)
        th2tf[th_name] = 'encoder/embedding/embeddings_{0}:0'.format(i)

    return th2tf


def theano_to_tensorflow_config(model_path):
    config = util.load_config(model_path)
    compat.fill_options(config)
    config['reload'] = None
    config['prior_model'] = None
    return argparse.Namespace(**config)


def theano_to_tensorflow_model(in_path, out_path):
    saved_model = np.load(in_path)
    config = theano_to_tensorflow_config(in_path)
    th2tf = construct_parameter_map(config)

    with tf.Session() as sess:
        model, saver = nmt.create_model(config, sess)
        seen = set()
        assign_ops = []
        for key in saved_model.keys():
            tf_name = th2tf[key]
            if tf_name is not None:
                assert tf_name not in seen
                seen.add(tf_name)
                tf_var = tf.get_default_graph().get_tensor_by_name(tf_name)
                if (sess.run(tf.shape(tf_var)) != saved_model[key].shape).any():
                    print "mismatch for", tf_name, key, saved_model[key].shape, sess.run(tf.shape(tf_var))
                assign_ops.append(tf.assign(tf_var, saved_model[key]))
            else:
                print "Not saving", key, "because no TF equivalent"
        sess.run(assign_ops)
        saver.save(sess, save_path=out_path)

        print "The following TF variables were not assigned (excluding Adam vars):"
        print "You should see only 'beta1_power', 'beta2_power' and 'time' variable listed"
        for tf_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if tf_var.name not in seen and 'Adam' not in tf_var.name:
                print tf_var.name


def tensorflow_to_theano_model(in_path, out_path):
    config = util.load_config(in_path)
    th2tf = construct_parameter_map(config)
    keys, values = zip(*th2tf.items())
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(in_path + '.meta')
        new_saver.restore(sess, in_path)
        params = {}
        for th_name, tf_name in th2tf.items():
            if tf_name is not None:
                try:
                    v = sess.run(tf.get_default_graph().get_tensor_by_name(tf_name))
                except:
                    print "Skipping {} because it was not found".format(tf_name)
                    continue
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
    opts.inn = os.path.abspath(opts.inn)
    opts.out = os.path.abspath(opts.out)
    
    if opts.from_theano:
        theano_to_tensorflow_model(opts.inn, opts.out)
    elif opts.from_tf:
        tensorflow_to_theano_model(opts.inn, opts.out)
    else:
        assert False
