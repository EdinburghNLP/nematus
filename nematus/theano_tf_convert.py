#!/usr/bin/env python3 

import argparse
import logging
import os
import sys

if __name__ == '__main__':
    # Configure the logging output. This needs to be done before the
    # tensorflow module is imported.
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

import numpy as np
import tensorflow as tf

from config import load_config_from_json_file
import model_loader
import rnn_model

def construct_parameter_map(config):
    def drt_tag(i):
        return "" if i == 0 else "_drt_{0}".format(i)

    def add_gru_variables(param_map, th_prefix, tf_prefix, drt_tag,
                          alt_names=False):
        for th_roots, tf_root in [[["U",  "U_nl"],  "state_to_gates"],
                                  [["Ux", "Ux_nl"], "state_to_proposal"],
                                  [["W",  "Wc"],    "input_to_gates"],
                                  [["Wx", "Wcx"],   "input_to_proposal"],
                                  [["b",  "b_nl"],  "gates_bias"],
                                  [["bx", "bx_nl"], "proposal_bias"]]:
            th_root = th_roots[1] if alt_names else th_roots[0]
            if drt_tag != "" and th_root.startswith("W"):
                # For deep transition, only the bottom GRU has external inputs.
                continue
            key = "{0}{1}{2}".format(th_prefix, th_root, drt_tag)
            val = "{0}{1}:0".format(tf_prefix, tf_root)
            param_map[key] = val

        for th_roots, tf_root in [[["U",  "U_nl"],  "gates_state_norm"],
                                  [["Ux", "Ux_nl"], "proposal_state_norm"],
                                  [["W",  "Wc"],    "gates_x_norm"],
                                  [["Wx", "Wcx"],   "proposal_x_norm"]]:
            th_root = th_roots[1] if alt_names else th_roots[0]
            if drt_tag != "" and th_root.startswith("W"):
                # For deep transition, only the bottom GRU has external inputs.
                continue
            key = "{0}{1}{2}_lnb".format(th_prefix, th_root, drt_tag)
            val = "{0}{1}/new_mean:0".format(tf_prefix, tf_root)
            param_map[key] = val
            key = "{0}{1}{2}_lns".format(th_prefix, th_root, drt_tag)
            val = "{0}{1}/new_std:0".format(tf_prefix, tf_root)
            param_map[key] = val

    th2tf = {
        # encoder/embedding
        'Wemb' : 'encoder/embedding/embeddings:0',

        # decoder/initial_state_constructor
        'ff_state_W' : 'decoder/initial_state_constructor/W:0',
        'ff_state_b' : 'decoder/initial_state_constructor/b:0',
        'ff_state_ln_b' : 'decoder/initial_state_constructor/new_mean:0',
        'ff_state_ln_s' : 'decoder/initial_state_constructor/new_std:0',

        # decoder/embedding
        'Wemb_dec' : 'decoder/embedding/embeddings:0',

        # decoder/base/attention
        'decoder_U_att' : 'decoder/base/attention/hidden_to_score:0',
        'decoder_W_comb_att' : 'decoder/base/attention/state_to_hidden:0',
        'decoder_W_comb_att_lnb' : 'decoder/base/attention/hidden_state_norm/new_mean:0',
        'decoder_W_comb_att_lns' : 'decoder/base/attention/hidden_state_norm/new_std:0',
        'decoder_Wc_att' : 'decoder/base/attention/context_to_hidden:0',
        'decoder_Wc_att_lnb' : 'decoder/base/attention/hidden_context_norm/new_mean:0',
        'decoder_Wc_att_lns' : 'decoder/base/attention/hidden_context_norm/new_std:0',
        'decoder_b_att' : 'decoder/base/attention/hidden_bias:0',

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

    # Add embedding variables for any additional factors.
    for i in range(1, len(config.dim_per_factor)):
        th_name = 'Wemb{0}'.format(i)
        th2tf[th_name] = 'encoder/embedding/embeddings_{0}:0'.format(i)

    # Add GRU variables for the encoder.
    for i in range(config.rnn_enc_depth):
        for j in range(config.rnn_enc_transition_depth):
            th_prefix_f = "encoder_" + ("" if i == 0 else "{0}_".format(i+1))
            tf_prefix_f = "encoder/forward-stack/level{0}/gru{1}/".format(i, j)
            th_prefix_b = "encoder_r_" + ("" if i == 0 else "{0}_".format(i+1))
            tf_prefix_b = "encoder/backward-stack/level{0}/gru{1}/".format(i, j)
            if i % 2:
                # The variable naming convention differs between the Theano and
                # Tensorflow versions: in the Theano version, encoder_<i> is
                # used for the i-th left-to-right encoder GRU, and encoder_r_<i>
                # is used for the i-th right-to-left one. In the Tensorflow
                # version, forward-stack/level0 is left-to-right and
                # backward-stack/level0 is right-to-left, but then the
                # directions alternate up the stack.  Flipping the th_prefixes
                # will map the GRU variables accordingly.
                th_prefix_f, th_prefix_b = th_prefix_b, th_prefix_f
            add_gru_variables(th2tf, th_prefix_f, tf_prefix_f, drt_tag(j))
            add_gru_variables(th2tf, th_prefix_b, tf_prefix_b, drt_tag(j))

    # Add GRU variables for the base level of the decoder.
    add_gru_variables(th2tf, "decoder_", "decoder/base/gru0/", "")
    for j in range(1, config.rnn_dec_base_transition_depth):
        tf_prefix = "decoder/base/gru{0}/".format(j)
        add_gru_variables(th2tf, "decoder_", tf_prefix, drt_tag(j-1),
                          alt_names=True)

    # Add GRU variables for the high levels of the decoder.
    for i in range(config.rnn_dec_depth-1):
        for j in range(config.rnn_dec_high_transition_depth):
            th_prefix = "decoder_{0}_".format(i+2)
            tf_prefix = "decoder/high/level{0}/gru{1}/".format(i, j)
            add_gru_variables(th2tf, th_prefix, tf_prefix, drt_tag(j))

    return th2tf


def theano_to_tensorflow_config(model_path):
    config = load_config_from_json_file(model_path)
    setattr(config, 'reload', None)
    setattr(config, 'prior_model', None)
    return config


def theano_to_tensorflow_model(in_path, out_path):
    saved_model = np.load(in_path)
    config = theano_to_tensorflow_config(in_path)
    th2tf = construct_parameter_map(config)

    with tf.Session() as sess:
        logging.info('Building model...')
        model = rnn_model.RNNModel(config)
        init = tf.zeros_initializer(dtype=tf.int32)
        global_step = tf.get_variable('time', [], initializer=init, trainable=False)
        saver = model_loader.init_or_restore_variables(config, sess)
        seen = set()
        assign_ops = []
        for th_name in list(saved_model.keys()):
            # ignore adam parameters
            if th_name.startswith('adam'):
                continue
            tf_name = th2tf[th_name]
            if tf_name is None:
                logging.info("Not saving {} because no TF " \
                             "equivalent".format(th_name))
                continue
            assert tf_name not in seen
            seen.add(tf_name)
            tf_var = tf.get_default_graph().get_tensor_by_name(tf_name)
            tf_shape = sess.run(tf.shape(tf_var))
            th_var = saved_model[th_name]
            th_shape = th_var.shape
            if list(tf_shape) != list(th_shape):
                logging.error("Shapes do not match for {} and " \
                              "{}.".format(tf_name, th_name))
                logging.error("Shape of {} is {}".format(tf_name, tf_shape))
                logging.error("Shape of {} is {}".format(th_name, th_shape))
                sys.exit(1)
            assign_ops.append(tf.assign(tf_var, th_var))
        sess.run(assign_ops)
        saver.save(sess, save_path=out_path)

        unassigned = []
        for tf_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if tf_var.name not in seen:
                unassigned.append(tf_var.name)
        logging.info("The following TF variables were not " \
                     "assigned: {}".format(" ".join(unassigned)))
        logging.info("You should see only the 'time' variable listed")


def tensorflow_to_theano_model(in_path, out_path):
    config = load_config_from_json_file(in_path)
    th2tf = construct_parameter_map(config)
    keys, values = list(zip(*list(th2tf.items())))
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(in_path + '.meta')
        new_saver.restore(sess, in_path)
        params = {}
        for th_name, tf_name in list(th2tf.items()):
            if tf_name is not None:
                try:
                    v = sess.run(tf.get_default_graph().get_tensor_by_name(tf_name))
                except:
                    logging.info("Skipping {} because it was not " \
                                 "found".format(tf_name))
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
    logging.info('Saved {} params to {}'.format(len(params), out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from_theano', action='store_true',
                       help="convert from Theano to TensorFlow format")
    group.add_argument('--from_tf', action='store_true',
                       help="convert from Tensorflow to Theano format")
    parser.add_argument('--in', type=str, required=True, metavar='PATH',
                        dest='inn', help="path to input model")
    parser.add_argument('--out', type=str, required=True, metavar='PATH',
                        help="path to output model")

    opts = parser.parse_args()
    opts.inn = os.path.abspath(opts.inn)
    opts.out = os.path.abspath(opts.out)

    if opts.from_theano:
        theano_to_tensorflow_model(opts.inn, opts.out)
    elif opts.from_tf:
        tensorflow_to_theano_model(opts.inn, opts.out)
    else:
        assert False
