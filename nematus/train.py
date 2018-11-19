#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import argparse
import collections
from datetime import datetime
import json
import os
import logging
import subprocess
import sys
import tempfile
import time

import numpy
import tensorflow as tf

from data_iterator import TextIterator, MultiTargetIterator
import exception
import inference
import model_loader
from model_updater import ModelUpdater
import rnn_model
import util


def load_data(config):
    logging.info('Reading data...')
    text_iterator = MultiTargetIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_langs=config.source_embedding_ids,
                        target_langs=config.target_embedding_ids,
                        source_dicts=config.source_dicts,
                        target_dicts=config.target_dicts,
                        embedding_map=config.embedding_map,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        source_vocab_sizes=config.source_vocab_sizes,
                        target_vocab_size=config.target_vocab_size,
                        skip_empty=True,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        use_factor=(config.factors > 1),
                        maxibatch_size=config.maxibatch_size,
                        token_batch_size=config.token_batch_size,
                        keep_data_in_memory=config.keep_train_set_in_memory)

    if config.validFreq and config.valid_source_dataset and config.valid_target_dataset:
        valid_text_iterator = MultiTargetIterator(
                            source=config.valid_source_dataset,
                            target=config.valid_target_dataset,
                            source_langs=config.source_embedding_ids,
                            target_langs=config.target_embedding_ids,
                            source_dicts=config.source_dicts,
                            target_dicts=config.target_dicts,
                            embedding_map=config.valid_embedding_map,
                            batch_size=config.valid_batch_size,
                            maxlen=config.maxlen,
                            source_vocab_sizes=config.source_vocab_sizes,
                            target_vocab_size=config.target_vocab_size,
                            shuffle_each_epoch=False,
                            sort_by_length=True,
                            use_factor=(config.factors > 1),
                            maxibatch_size=config.maxibatch_size,
                            token_batch_size=config.valid_token_batch_size)
    else:
        logging.info('no validation set loaded')
        valid_text_iterator = None
    logging.info('Done')
    return text_iterator, valid_text_iterator


def train(config, sess):
    assert (config.prior_model != None and (tf.train.checkpoint_exists(os.path.abspath(config.prior_model))) or (config.map_decay_c==0.0)), \
    "MAP training requires a prior model file: Use command-line option --prior_model"

    # Construct the graph, with one model replica per GPU

    num_gpus = len(util.get_available_gpus())
    num_replicas = max(1, num_gpus)

    logging.info('Building model...')
    replicas = []
    for i in range(num_replicas):
        device_type = "GPU" if num_gpus > 0 else "CPU"
        device_spec = tf.DeviceSpec(device_type=device_type, device_index=i)
        with tf.device(device_spec):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(i>0)):
                replicas.append(rnn_model.RNNModel(config))

    if config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                           beta1=config.adam_beta1,
                                           beta2=config.adam_beta2,
                                           epsilon=config.adam_epsilon)
    else:
        logging.error('No valid optimizer defined: {}'.format(config.optimizer))
        sys.exit(1)

    init = tf.zeros_initializer(dtype=tf.int32)
    global_step = tf.get_variable('time', [], initializer=init, trainable=False)

    if config.summaryFreq:
        summary_dir = (config.summary_dir if config.summary_dir is not None
                       else os.path.abspath(os.path.dirname(config.saveto)))
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
    else:
        writer = None

    updater = ModelUpdater(config, num_gpus, replicas, optimizer, global_step,
                           writer)

    saver, progress = model_loader.init_or_restore_variables(
        config, sess, train=True)

    global_step.load(progress.uidx, sess)

    # Use an InferenceModelSet to abstract over model types for sampling and
    # beam search. Multi-GPU sampling and beam search are not currently
    # supported, so we just use the first replica.
    model_set = inference.InferenceModelSet([replicas[0]], [config])

    #save model options
    config_as_dict = collections.OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('%s.json' % config.saveto, 'wb'), indent=2)

    text_iterator, valid_text_iterator = load_data(config)
    _, _, num_to_source, num_to_target = util.load_dictionaries(config)
    num_to_multi_source, vocab_offsets = util.combine_source_dicts(
        num_to_source, config.factors, config.source_vocab_sizes)
    total_loss = 0.
    n_sents, n_words = 0, 0
    last_time = time.time()
    logging.info("Initial uidx={}".format(progress.uidx))
    for progress.eidx in xrange(progress.eidx, config.max_epochs):
        logging.info('Starting epoch {0}'.format(progress.eidx))
        for source_sents, target_sents, target_lang in text_iterator:
            if len(source_sents[0][0]) != config.factors:
                logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(config.factors, len(source_sents[0][0])))
                sys.exit(1)
            x_in, x_mask_in, y_in, y_mask_in = util.prepare_data(
                source_sents, target_sents, config.factors, maxlen=None)
            if x_in is None:
                logging.info('Minibatch with zero sample under length {0}'.format(config.maxlen))
                continue
            write_summary_for_this_batch = config.summaryFreq and ((progress.uidx % config.summaryFreq == 0) or (config.finish_after and progress.uidx % config.finish_after == 0))
            (factors, seqLen, batch_size) = x_in.shape

            loss = updater.update(sess, x_in, x_mask_in, y_in, y_mask_in,
                                  target_lang, write_summary_for_this_batch)
            total_loss += loss
            n_sents += batch_size
            n_words += int(numpy.sum(y_mask_in))
            progress.uidx += 1

            if config.dispFreq and progress.uidx % config.dispFreq == 0:
                duration = time.time() - last_time
                disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                logging.info('{0} Epoch: {1} Update: {2} Loss/word: {3} Words/sec: {4} Sents/sec: {5}'.format(disp_time, progress.eidx, progress.uidx, total_loss/n_words, n_words/duration, n_sents/duration))
                last_time = time.time()
                total_loss = 0.
                n_sents = 0
                n_words = 0

            if config.sampleFreq and progress.uidx % config.sampleFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:, :10]
                samples = model_set.sample(sess, x_small, x_mask_small,
                                           target_lang)
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    source = util.factoredseq2words(xx, num_to_multi_source,
                                                    vocab_offsets)
                    target = util.seq2words(yy, num_to_target[target_lang])
                    sample = util.seq2words(ss, num_to_target[target_lang])
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    logging.info('SAMPLE: {}'.format(sample))

            if config.beamFreq and progress.uidx % config.beamFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:,:10]
                samples = model_set.beam_search(sess, x_small, x_mask_small,
                                                target_lang, config.beam_size)
                # samples is a list with shape batch x beam x len
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    source = util.factoredseq2words(xx, num_to_multi_source,
                                                    vocab_offsets)
                    target = util.seq2words(yy, num_to_target[target_lang])
                    logging.info('SOURCE: {}'.format(source))
                    logging.info('TARGET: {}'.format(target))
                    for i, (sample_seq, cost) in enumerate(ss):
                        sample = util.seq2words(sample_seq,
                                                num_to_target[target_lang])
                        msg = 'SAMPLE {}: {} Cost/Len/Avg {}/{}/{}'.format(
                            i, sample, cost, len(sample), cost/len(sample))
                        logging.info(msg)

            if config.validFreq and progress.uidx % config.validFreq == 0:
                costs = validate(config, sess, valid_text_iterator, replicas[0])
                # validation loss is mean of normalized sentence log probs
                valid_loss = sum(costs) / len(costs)
                if (len(progress.history_errs) == 0 or
                    valid_loss < min(progress.history_errs)):
                    progress.history_errs.append(valid_loss)
                    progress.bad_counter = 0
                    saver.save(sess, save_path=config.saveto)
                    progress_path = '{0}.progress.json'.format(config.saveto)
                    progress.save_to_json(progress_path)
                else:
                    progress.history_errs.append(valid_loss)
                    progress.bad_counter += 1
                    if progress.bad_counter > config.patience:
                        logging.info('Early Stop!')
                        progress.estop = True
                        break
                if config.valid_script is not None:
                    score = validate_with_script(sess, replicas[0], config)
                    need_to_save = (score is not None and
                        (len(progress.valid_script_scores) == 0 or
                         score > max(progress.valid_script_scores)))
                    if score is None:
                        score = 0.0  # ensure a valid value is written
                    progress.valid_script_scores.append(score)
                    if need_to_save:
                        save_path = config.saveto + ".best-valid-script"
                        saver.save(sess, save_path=save_path)
                        progress_path = '{}.progress.json'.format(save_path)
                        progress.save_to_json(progress_path)

            if config.saveFreq and progress.uidx % config.saveFreq == 0:
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)

            if config.finish_after and progress.uidx % config.finish_after == 0:
                logging.info("Maximum number of updates reached")
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress.estop=True
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)
                break
        if progress.estop:
            break


def validate_with_script(sess, model, config):
    if config.valid_script == None:
        return None
    logging.info('Starting external validation.')
    out = tempfile.NamedTemporaryFile()
    inference.translate_file(input_file=open(config.valid_source_dataset),
                             output_file=out,
                             session=sess,
                             models=[model],
                             configs=[config],
                             beam_size=config.beam_size,
                             minibatch_size=config.valid_batch_size,
                             normalization_alpha=1.0)
    out.flush()
    args = [config.valid_script, out.name]
    proc = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if len(stderr) > 0:
        logging.info("Validation script wrote the following to standard "
                     "error:\n" + stderr)
    if proc.returncode != 0:
        logging.warning("Validation script failed (returned exit status of "
                        "{}).".format(proc.returncode))
        return None
    try:
        score = float(stdout.split()[0])
    except:
        logging.warning("Validation script output does not look like a score: "
                        "{}".format(stdout))
        return None
    logging.info("Validation script score: {}".format(score))
    return score


# TODO Support for multiple GPUs
def calc_loss_per_sentence(config, sess, text_iterator, model,
                           normalization_alpha=0):
    losses = []
    for x_v, y_v, target_lang in text_iterator:
        if len(x_v[0][0]) != config.factors:
            logging.error('Mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(config.factors, len(x_v[0][0])))
            sys.exit(1)
        x, x_mask, y, y_mask = util.prepare_data(x_v, y_v, config.factors,
                                                 maxlen=None)
        feeds = {model.inputs.x: x,
                 model.inputs.x_mask: x_mask,
                 model.inputs.y: y,
                 model.inputs.y_mask: y_mask,
                 model.inputs.target_lang_id: target_lang,
                 model.inputs.training: False}
        loss_per_sentence = sess.run(model.loss_per_sentence, feed_dict=feeds)

        # normalize scores according to output length
        if normalization_alpha:
            adjusted_lengths = numpy.array([numpy.count_nonzero(s) ** normalization_alpha for s in y_mask.T])
            loss_per_sentence /= adjusted_lengths

        losses += list(loss_per_sentence)
        logging.info( "Seen {0}".format(len(losses)))
    return losses


def validate(config, sess, text_iterator, model, normalization_alpha=0):
    losses = calc_loss_per_sentence(config, sess, text_iterator, model,
                                    normalization_alpha)
    num_sents = len(losses)
    total_loss = sum(losses)
    logging.info('Validation loss (AVG/SUM/N_SENT): {0} {1} {2}'.format(
        total_loss/num_sents, total_loss, num_sents))
    return losses


def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')

    data.add_argument('--source_dataset', type=str, metavar='PATH', 
                         help="parallel training corpus (source)")
    data.add_argument('--target_dataset', type=str, metavar='PATH', 
                         help="parallel training corpus (target)")
    # parallel training corpus (source and target). Hidden option for backward compatibility
    data.add_argument('--datasets', type=str, metavar='PATH', nargs=2,
                         help=argparse.SUPPRESS)
    data_dicts = data.add_mutually_exclusive_group(required=True)
    data_dicts.add_argument('--dictionaries', type=str, metavar='PATH', nargs="+",
                          help="network vocabularies (one per source factor, plus target vocabulary)")
    data_dicts.add_argument('--source_dicts', type=str, nargs='+', metavar='STR',
                          help="list of source language vocabularies, one per factor or one per (embedding, factor) pair if using multiple source embeddings (use embedding-major order)")
    data.add_argument('--source_embedding_ids', type=str, default=["src"], nargs='+', metavar='STR',
                         help="list of source embedding identifiers, e.g. --source_embedding_ids en de nl ro (default: src)")
    data.add_argument('--target_dicts', type=str, nargs='+', metavar='STR',
                         help="list of target language vocabularies (one per embedding)")
    data.add_argument('--target_embedding_ids', type=str, default=["tgt"], nargs='+', metavar='STR',
                         help="list of target language identifiers, e.g. --target_embedding_ids en de nl ro (default: tgt)")
    data.add_argument('--embedding_map', type=str, default=None, metavar='STR',
                         help="file listing source-target embedding identifiers for each sentence pair in the training corpus")
    data.add_argument('--valid_embedding_map', type=str, default=None, metavar='STR',
                         help="file listing source-target embedding identifiers for each sentence pair in the validation set")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--model', '--saveto', type=str, default='model', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--reload', type=str, default=None, metavar='PATH',
                         help="load existing model from this path. Set to \"latest_checkpoint\" to reload the latest checkpoint in the same directory of --model")
    data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                         help="don't reload training progress (only used if --reload is enabled)")
    data.add_argument('--summary_dir', type=str, required=False, metavar='PATH', 
                         help="directory for saving summaries (default: same directory as the --model file)")
    data.add_argument('--summaryFreq', type=int, default=0, metavar='INT',
                         help="Save summaries after INT updates, if 0 do not save summaries (default: %(default)s)")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--embedding_size', '--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--state_size', '--dim', type=int, default=1000, metavar='INT',
                         help="hidden state size (default: %(default)s)")
    network.add_argument('--source_vocab_sizes', '--n_words_src', type=int, default=None, nargs='+', metavar='INT',
                         help="source vocabulary sizes (default: %(default)s)")
    network.add_argument('--target_vocab_size', '--n_words', type=int, default=None, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")
    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")

    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--enc_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',
                         help="number of decoder layers (default: %(default)s)")
    network.add_argument('--dec_base_recurrence_transition_depth', type=int, default=2, metavar='INT',
                         help="number of GRU transition operations applied in the first layer of the decoder. Minimum is 2.  (Only applies to gru_cond). (default: %(default)s)")
    network.add_argument('--dec_high_recurrence_transition_depth', type=int, default=1, metavar='INT',
                         help="number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru). (default: %(default)s)")
    network.add_argument('--dec_deep_context', action='store_true',
                         help="pass context vector (from first layer) to deep decoder layers")
    network.add_argument('--use_dropout', action="store_true",
                         help="use dropout layer (default: %(default)s)")
    network.add_argument('--dropout_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for input embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_source', type=float, default=0.0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0.0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")
    network.add_argument('--use_layer_norm', '--layer_normalisation', action="store_true", dest="use_layer_norm",
                         help="Set to use layer normalization in encoder and decoder")
    network.add_argument('--tie_encoder_decoder_embeddings', action="store_true", dest="tie_encoder_decoder_embeddings",
                         help="tie the input embeddings of the encoder and the decoder (first factor only). Source and target vocabulary size must be the same")
    network.add_argument('--tie_decoder_embeddings', action="store_true", dest="tie_decoder_embeddings",
                         help="tie the input embeddings of the decoder with the softmax output embeddings")
    network.add_argument('--output_hidden_activation', type=str, default='tanh',
                         choices=['tanh', 'relu', 'prelu', 'linear'],
                         help='activation function in hidden layer of the output network (default: %(default)s)')
    network.add_argument('--softmax_mixture_size', type=int, default=1, metavar="INT",
                         help="number of softmax components to use (default: %(default)s)")

    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length for training and validation (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--token_batch_size', type=int, default=0, metavar='INT',
                          help="minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, batch_size only affects sorting by length. (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=10000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--decay_c', type=float, default=0.0, metavar='FLOAT',
                         help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0.0, metavar='FLOAT',
                         help="MAP-L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--prior_model', type=str, metavar='PATH',
                         help="Prior model for MAP-L2 regularization. Unless using \"--reload\", this will also be used for initialization.")
    training.add_argument('--clip_c', type=float, default=1.0, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--label_smoothing', type=float, default=0.0, metavar='FLOAT',
                         help="label smoothing (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--keep_train_set_in_memory', action="store_true", 
                         help="Keep training dataset lines stores in RAM during training")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument(
        '--optimizer', type=str, default="adam", choices=['adam'],
        help="optimizer (default: %(default)s)")
    training.add_argument(
        '--learning_rate', '--lrate', type=float, default=0.0001,
        metavar='FLOAT',
        help="learning rate (default: %(default)s)")
    training.add_argument(
        '--adam_beta1', type=float, default=0.9, metavar='FLOAT',
        help='exponential decay rate for the first moment estimates ' \
             '(default: %(default)s)')
    training.add_argument(
        '--adam_beta2', type=float, default=0.999, metavar='FLOAT',
        help='exponential decay rate for the second moment estimates ' \
             '(default: %(default)s)')
    training.add_argument(
        '--adam_epsilon', type=float, default=1e-08, metavar='FLOAT', \
        help='constant for numerical stability (default: %(default)s)')

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_source_dataset', type=str, default=None, metavar='PATH', 
                         help="source validation corpus (default: %(default)s)")
    validation.add_argument('--valid_target_dataset', type=str, default=None, metavar='PATH',
                         help="target validation corpus (default: %(default)s)")
    # parallel validation corpus (source and target). Hidden option for backward compatibility
    validation.add_argument('--valid_datasets', type=str, default=None, metavar='PATH', nargs=2,
                         help=argparse.SUPPRESS)
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validation minibatch size (default: %(default)s)")
    training.add_argument('--valid_token_batch_size', type=int, default=0, metavar='INT',
                          help="validation minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, valid_batch_size only affects sorting by length. (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--valid_script', type=str, default=None, metavar='PATH',
                         help="path to script for external validation (default: %(default)s). The script will be passed an argument specifying the path of a file that contains translations of the source validation corpus. It must write a single score to standard output.")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")
    display.add_argument('--beamFreq', type=int, default=10000, metavar='INT',
                         help="display some beam_search samples after INT updates (default: %(default)s)")
    display.add_argument('--beam_size', type=int, default=12, metavar='INT',
                         help="size of the beam (default: %(default)s)")

    translate = parser.add_argument_group('translate parameters')
    translate.add_argument('--no_normalize', action='store_false', dest='normalize',
                            help="Cost of sentences will not be normalized by length")
    translate.add_argument('--n_best', action='store_true', dest='n_best',
                            help="Print full beam")
    translate.add_argument('--translation_maxlen', type=int, default=200, metavar='INT',
                         help="Maximum length of translation output sentence (default: %(default)s)")
    config = parser.parse_args()

    # allow "--datasets" for backward compatibility
    if config.datasets:
        if config.source_dataset or config.target_dataset:
            logging.error('argument clash: --datasets is mutually exclusive with --source_dataset and --target_dataset')
            sys.exit(1)
        else:
            config.source_dataset = config.datasets[0]
            config.target_dataset = config.datasets[1]
    elif not config.source_dataset:
        logging.error('--source_dataset is required')
        sys.exit(1)
    elif not config.target_dataset:
        logging.error('--target_dataset is required')
        sys.exit(1)

    # allow "--valid_datasets" for backward compatibility
    if config.valid_datasets:
        if config.valid_source_dataset or config.valid_target_dataset:
            logging.error('argument clash: --valid_datasets is mutually exclusive with --valid_source_dataset and --valid_target_dataset')
            sys.exit(1)
        else:
            config.valid_source_dataset = config.valid_datasets[0]
            config.valid_target_dataset = config.valid_datasets[1]

    # check factor-related options are consistent

    if config.dim_per_factor == None:
        if config.factors == 1:
            config.dim_per_factor = [config.embedding_size]
        else:
            logging.error('if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    if len(config.dim_per_factor) != config.factors:
        logging.error('mismatch between \'--factors\' ({0}) and \'--dim_per_factor\' ({1} entries)\n'.format(config.factors, len(config.dim_per_factor)))
        sys.exit(1)

    if sum(config.dim_per_factor) != config.embedding_size:
        logging.error('mismatch between \'--embedding_size\' ({0}) and \'--dim_per_factor\' (sums to {1})\n'.format(config.embedding_size, sum(config.dim_per_factor)))
        sys.exit(1)

    # allow --dictionaries for backward compatibility
    if config.dictionaries != None:
        config.source_dicts = config.dictionaries[:config.factors]
        config.target_dict = config.dictionaries[-1]
    else:
        if len(config.target_dicts) == 0:
            logging.error('\'--target_dicts\' is required if using \'--source_dicts\'\n')
            sys.exit(1)

    # check embedding-related options are consistent

    if (len(config.source_embedding_ids) > 1 or
        len(config.target_embedding_ids) > 1):
        if config.embedding_map == None:
            logging.error('\'--embedding_map\' is required if using more than one source or target embedding\n')
            sys.exit(1)

    if len(config.source_dicts) != (len(config.source_embedding_ids) * config.factors):
        logging.error('\'--source_dicts\' must specify one dictionary per source factor, for each source embedding\n')
        sys.exit(1)

    if len(config.target_dicts) != len(config.target_embedding_ids):
        logging.error('\'--target_dicts\' must specify one dictionary per target embedding\n')
        sys.exit(1)

    # determine target_embedding_size
    if config.tie_encoder_decoder_embeddings:
        config.target_embedding_size = config.dim_per_factor[0]
    else:
        config.target_embedding_size = config.embedding_size

    if config.source_vocab_sizes != None or config.target_vocab_size != None:
        logging.error('\'--source_vocab_sizes\' and \'--target_vocab_size\' are not currently supported in this branch\n')
        sys.exit(1)

#    # set vocabulary sizes
#    vocab_sizes = []
#    if config.source_vocab_sizes == None:
#        vocab_sizes = [-1] * config.factors
#    elif len(config.source_vocab_sizes) == config.factors:
#        vocab_sizes = config.source_vocab_sizes
#    elif len(config.source_vocab_sizes) < config.factors:
#        num_missing = config.factors - len(config.source_vocab_sizes)
#        vocab_sizes += config.source_vocab_sizes + [-1] * num_missing
#    else:
#        logging.error('too many values supplied to \'--source_vocab_sizes\' option (expected one per factor = {0})'.format(config.factors))
#        sys.exit(1)
#    if config.target_vocab_size == -1:
#        vocab_sizes.append(-1)
#    else:
#        vocab_sizes.append(config.target_vocab_size)
#
#    # for unspecified vocabulary sizes, determine sizes from vocabulary dictionaries
#    for i, vocab_size in enumerate(vocab_sizes):
#        if vocab_size >= 0:
#            continue
#        try:
#            d = load_dict(config.dictionaries[i])
#        except:
#            logging.error('failed to determine vocabulary size from file: {0}'.format(config.dictionaries[i]))
#        vocab_sizes[i] = max(d.values()) + 1
#
#    config.source_dicts = config.dictionaries[:config.factors]
#    config.source_vocab_sizes = vocab_sizes[:-1]
#    config.target_dicts = config.dictionaries[config.factors:]
#    config.target_vocab_size = vocab_sizes[-1]

    def get_vocab_size_or_die(path):
        try:
            d = util.load_dict(path)
        except:
            logging.error('failed to determine vocabulary size from file: {0}'.format(path))
        return max(d.values()) + 1

    config.source_vocab_sizes = []
    for i in range(len(config.source_embedding_ids)):
        for j in range(config.factors):
            path = config.source_dicts[i * config.factors + j]
            vocab_size = get_vocab_size_or_die(path)
            config.source_vocab_sizes.append(vocab_size)

    config.target_vocab_size = 0
    for i in range(len(config.target_embedding_ids)):
        path = config.target_dicts[i]
        vocab_size = get_vocab_size_or_die(path)
        config.target_vocab_size = max(config.target_vocab_size, vocab_size)

    # set the model version
    config.model_version = 0.2
    config.theano_compat = False

    return config


if __name__ == "__main__":
    # Start logging.
    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Parse command-line arguments.
    config = parse_args()
    logging.info(config)

    # Create the TensorFlow session.
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True

    # Train.
    with tf.Session(config=tf_config) as sess:
        train(config, sess)
