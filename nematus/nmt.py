#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Build a neural machine translation model with soft attention
'''
import os
import sys
import logging
import time
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim

from threading import Thread
import queue
from datetime import datetime
from collections import OrderedDict

from layers import *
from data_iterator import TextIterator

from model import *
from util import *
import training_progress
import exception
import compat

def create_model(config, sess, ensemble_scope=None, train=False):
    logging.info('Building model...')
    model = StandardModel(config)

    # Construct a mapping between saved variable names and names in the current
    # scope. There are two reasons why names might be different:
    #
    #   1. This model is part of an ensemble, in which case a model-specific
    #       name scope will be active.
    #
    #   2. The saved model is from an old version of Nematus (before deep model
    #        support was added) and uses a different variable naming scheme
    #        for the GRUs.
    variables = slim.get_variables_to_restore()
    var_map = {}
    for v in variables:
        name = v.name.split(':')[0]
        if ensemble_scope == None:
            saved_name = name
        elif v.name.startswith(ensemble_scope):
            saved_name = name[len(ensemble_scope):]
        else: # v belongs to a different model in the ensemble.
            continue
        if config.model_version == 0.1:
            # Backwards compatibility with the old variable naming scheme.
            saved_name = compat.revert_variable_name(saved_name, 0.1)
        var_map[saved_name] = v
    saver = tf.train.Saver(var_map, max_to_keep=None)

    # compute reload model filename
    reload_filename = None
    if config.reload == 'latest_checkpoint':
        checkpoint_dir = os.path.dirname(config.saveto)
        reload_filename = tf.train.latest_checkpoint(checkpoint_dir)
        if reload_filename != None:
            if (os.path.basename(reload_filename).rsplit('-', 1)[0] !=
                os.path.basename(config.saveto)):
                logging.error("Mismatching model filename found in the same directory while reloading from the latest checkpoint")
                sys.exit(1)
            logging.info('Latest checkpoint found in directory ' + os.path.abspath(checkpoint_dir))
    elif config.reload != None:
        reload_filename = config.reload
    if (reload_filename == None) and (config.prior_model != None):
        logging.info('Initializing model parameters from prior')
        reload_filename = config.prior_model

    # initialize or reload training progress
    if train:
        progress = training_progress.TrainingProgress()
        progress.bad_counter = 0
        progress.uidx = 0
        progress.eidx = 0
        progress.estop = False
        progress.history_errs = []
        if reload_filename and config.reload_training_progress:
            path = reload_filename + '.progress.json'
            if os.path.exists(path):
                logging.info('Reloading training progress')
                progress.load_from_json(path)
                if (progress.estop == True or
                    progress.eidx > config.max_epochs or
                    progress.uidx >= config.finish_after):
                    logging.warning('Training is already complete. Disable reloading of training progress (--no_reload_training_progress) or remove or modify progress file (%s) to train anyway.' % path)
                    sys.exit(0)

    # load prior model
    if train and config.prior_model != None:
        load_prior(config, sess, saver)
    
    # initialize or restore model
    if reload_filename == None:
        logging.info('Initializing model parameters from scratch...')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    else:
        logging.info('Loading model parameters from file ' + os.path.abspath(reload_filename))
        saver.restore(sess, os.path.abspath(reload_filename))
        if train:
            # The global step is currently recorded in two places:
            #   1. model.t, a tf.Variable read and updated by the optimizer
            #   2. progress.uidx, a Python integer updated by train()
            # We reset model.t to the value recorded in progress to allow the
            # value to be controlled by the user (either explicitly by
            # configuring the value in the progress file or implicitly by using
            # --no_reload_training_progress).
            model.reset_global_step(progress.uidx, sess)

    logging.info('Done')

    if train:
        return model, saver, progress
    else:
        return model, saver

def load_prior(config, sess, saver):
     logging.info('Loading prior model parameters from file ' + os.path.abspath(config.prior_model))
     saver.restore(sess, os.path.abspath(config.prior_model))
     
     # fill prior variables with the loaded values
     prior_variables = tf.get_collection_ref('prior_variables')
     prior_variables_dict = dict([(v.name, v) for v in prior_variables])
     assign_tensors = []
     with tf.name_scope('prior'):
         for v in tf.trainable_variables():
             prior_name = 'loss/prior/'+v.name
             prior_variable = prior_variables_dict[prior_name]
             assign_tensors.append(prior_variable.assign(v))
     tf.variables_initializer(prior_variables)
     sess.run(assign_tensors)

def load_data(config):
    logging.info('Reading data...')
    text_iterator = TextIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_dicts=config.source_dicts,
                        target_dict=config.target_dict,
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
        valid_text_iterator = TextIterator(
                            source=config.valid_source_dataset,
                            target=config.valid_target_dataset,
                            source_dicts=config.source_dicts,
                            target_dict=config.target_dict,
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

def load_dictionaries(config):
    source_to_num = [load_dict(d) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target

def read_all_lines(config, sentences):
    source_to_num, _, _, _ = load_dictionaries(config)
    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            if config.factors == 1:
                w = [source_to_num[0][w] if w in source_to_num[0] else 1]
            else:
                w = [source_to_num[i][f] if f in source_to_num[i] else 1
                                         for (i,f) in enumerate(w.split('|'))]
                if len(w) != config.factors:
                    raise exception.Error(
                        'Expected {0} factors, but input word has {1}\n'.format(
                            config.factors, len(w)))
            line.append(w)
        lines.append(line)
    lines = numpy.array(lines)
    lengths = numpy.array(map(lambda l: len(l), lines))
    lengths = numpy.array(lengths)
    idxs = lengths.argsort()
    lines = lines[idxs]

    #merge into batches
    batches = []
    for i in range(0, len(lines), config.valid_batch_size):
        batch = lines[i:i+config.valid_batch_size]
        batches.append(batch)

    return batches, idxs


def train(config, sess):
    assert config.prior_model != None and (tf.train.checkpoint_exists(os.path.abspath(config.prior_model))) or (config.map_decay_c==0.0), \
        "MAP training requires a prior model file: Use command-line option --prior_model"

    model, saver, progress = create_model(config, sess, train=True)

    x,x_mask,y,y_mask,training = model.get_score_inputs()
    apply_grads = model.get_apply_grads()
    t = model.get_global_step()
    loss_per_sentence = model.get_loss()
    objective = model.get_objective()

    if config.summaryFreq:
        summary_dir = config.summary_dir if (config.summary_dir != None) else (os.path.abspath(os.path.dirname(config.saveto)))
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
    tf.summary.scalar(name='mean_cost', tensor=objective)
    tf.summary.scalar(name='t', tensor=t)
    merged = tf.summary.merge_all()

    #save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('%s.json' % config.saveto, 'w'), indent=2)

    text_iterator, valid_text_iterator = load_data(config)
    _, _, num_to_source, num_to_target = load_dictionaries(config)
    total_loss = 0.
    n_sents, n_words = 0, 0
    last_time = time.time()
    logging.info("Initial uidx={}".format(progress.uidx))
    for progress.eidx in range(progress.eidx, config.max_epochs):
        logging.info('Starting epoch {0}'.format(progress.eidx))
        for source_sents, target_sents in text_iterator:
            if len(source_sents[0][0]) != config.factors:
                logging.error('Mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(config.factors, len(source_sents[0][0])))
                sys.exit(1)
            x_in, x_mask_in, y_in, y_mask_in = prepare_data(source_sents, target_sents, maxlen=None)
            if x_in is None:
                logging.info('Minibatch with zero sample under length {0}'.format(config.maxlen))
                continue
            write_summary_for_this_batch = config.summaryFreq and ((progress.uidx % config.summaryFreq == 0) or (config.finish_after and progress.uidx % config.finish_after == 0))
            (factors, seqLen, batch_size) = x_in.shape
            inn = {x:x_in, y:y_in, x_mask:x_mask_in, y_mask:y_mask_in, training:True}
            out = [t, apply_grads, objective]
            if write_summary_for_this_batch:
                out += [merged]
            out_values = sess.run(out, feed_dict=inn)
            objective_value = out_values[2]
            total_loss += objective_value*batch_size
            n_sents += batch_size
            n_words += int(numpy.sum(y_mask_in))
            progress.uidx += 1

            if write_summary_for_this_batch:
                writer.add_summary(out_values[3], out_values[0])

            if config.dispFreq and progress.uidx % config.dispFreq == 0:
                duration = time.time() - last_time
                disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                logging.info('{0} Epoch: {1} Update: {2} Loss/word: {3} Words/sec: {4} Sents/sec: {5}'.format(disp_time, progress.eidx, progress.uidx, total_loss/n_words, n_words/duration, n_sents/duration))
                last_time = time.time()
                total_loss = 0.
                n_sents = 0
                n_words = 0

            if config.saveFreq and progress.uidx % config.saveFreq == 0:
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)

            if config.sampleFreq and progress.uidx % config.sampleFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:, :10]
                samples = model.sample(sess, x_small, x_mask_small)
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    logging.info('SOURCE: {0}'.format(factoredseq2words(xx, num_to_source)))
                    logging.info('TARGET: {0}'.format(seq2words(yy, num_to_target)))
                    logging.info('SAMPLE: {0}'.format(seq2words(ss, num_to_target)))

            if config.beamFreq and progress.uidx % config.beamFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :, :10], x_mask_in[:, :10], y_in[:,:10]
                samples = model.beam_search(sess, x_small, x_mask_small, config.beam_size)
                # samples is a list with shape batch x beam x len
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    logging.info('SOURCE: {0}'.format(factoredseq2words(xx, num_to_source)))
                    logging.info('TARGET: {0}'.format(seq2words(yy, num_to_target)))
                    for i, (sample, cost) in enumerate(ss):
                        logging.info('SAMPLE {0}: {1} Cost/Len/Avg {2}/{3}/{4}'.format(i, seq2words(sample, num_to_target), cost, len(sample), cost/len(sample)))

            if config.validFreq and progress.uidx % config.validFreq == 0:
                costs = validate(config, sess, valid_text_iterator, model)
                # validation loss is mean of normalized sentence log probs
                valid_loss = sum(costs) / len(costs)
                if (len(progress.history_errs) == 0 or
                    valid_loss < min(progress.history_errs)):
                    progress.bad_counter = 0
                    saver.save(sess, save_path=config.saveto)
                    progress_path = '{0}.progress.json'.format(config.saveto)
                    progress.save_to_json(progress_path)
                else:
                    progress.bad_counter += 1
                    if progress.bad_counter > config.patience:
                        logging.info('Early Stop!')
                        progress.estop = True
                        break
                progress.history_errs.append(valid_loss)

            if config.finish_after and progress.uidx % config.finish_after == 0:
                logging.info("Maximum number of updates reached")
                saver.save(sess, save_path=config.saveto, global_step=progress.uidx)
                progress.estop=True
                progress_path = '{0}-{1}.progress.json'.format(config.saveto, progress.uidx)
                progress.save_to_json(progress_path)
                break
        if progress.estop:
            break

def translate(config, sess):
    model, saver = create_model(config, sess)
    start_time = time.time()
    _, _, _, num_to_target = load_dictionaries(config)
    logging.info("NOTE: Length of translations is capped to {}".format(config.translation_maxlen))

    n_sent = 0
    try:
        batches, idxs = read_all_lines(config, open(config.valid_source_dataset, 'r').readlines())
    except exception.Error as x:
        logging.error(x.msg)
        sys.exit(1)
    in_queue, out_queue = Queue(), Queue()
    model._get_beam_search_outputs(config.beam_size)
    
    def translate_worker(in_queue, out_queue, model, sess, config):
        while True:
            job = in_queue.get()
            if job is None:
                break
            idx, x = job
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = prepare_data(x, y_dummy, maxlen=None)
            try:
                samples = model.beam_search(sess, x, x_mask, config.beam_size)
                out_queue.put((idx, samples))
            except:
                in_queue.put(job)

    threads = [None] * config.n_threads
    for i in xrange(config.n_threads):
        threads[i] = Thread(
                        target=translate_worker,
                        args=(in_queue, out_queue, model, sess, config))
        threads[i].deamon = True
        threads[i].start()

    for i, batch in enumerate(batches):
        in_queue.put((i,batch))
    outputs = [None]*len(batches)
    for _ in range(len(batches)):
        i, samples = out_queue.get()
        outputs[i] = list(samples)
        n_sent += len(samples)
        logging.info('Translated {} sents'.format(n_sent))
    for _ in range(config.n_threads):
        in_queue.put(None)
    outputs = [beam for batch in outputs for beam in batch]
    outputs = numpy.array(outputs, dtype=numpy.object)
    outputs = outputs[idxs.argsort()]

    for beam in outputs:
        if config.normalize:
            beam = map(lambda sent_cost: (sent_cost[0], sent_cost[1]/len(sent_cost[0])), beam)
        beam = sorted(beam, key=lambda sent_cost: sent_cost[1])
        if config.n_best:
            for sent, cost in beam:
                print(seq2words(sent, num_to_target), '[%f]' % cost)
        else:
            best_hypo, cost = beam[0]
            print(seq2words(best_hypo, num_to_target))
    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(n_sent, duration, n_sent/duration))


def validate(config, sess, valid_text_iterator, model, normalization_alpha=0):
    costs = []
    total_loss = 0.
    total_seen = 0
    x,x_mask,y,y_mask,training = model.get_score_inputs()
    loss_per_sentence = model.get_loss()
    for x_v, y_v in valid_text_iterator:
        if len(x_v[0][0]) != config.factors:
            logging.error('Mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(config.factors, len(x_v[0][0])))
            sys.exit(1)
        x_v_in, x_v_mask_in, y_v_in, y_v_mask_in = prepare_data(x_v, y_v, maxlen=None)
        feeds = {x:x_v_in, x_mask:x_v_mask_in, y:y_v_in, y_mask:y_v_mask_in, training:False}
        loss_per_sentence_out = sess.run(loss_per_sentence, feed_dict=feeds)

        # normalize scores according to output length
        if normalization_alpha:
            adjusted_lengths = numpy.array([numpy.count_nonzero(s) ** normalization_alpha for s in y_v_mask_in.T])
            loss_per_sentence_out /= adjusted_lengths

        total_loss += loss_per_sentence_out.sum()
        total_seen += x_v_in.shape[2]
        costs += list(loss_per_sentence_out)
        logging.info( "Seen {0}".format(total_seen))
    logging.info('Validation loss (AVG/SUM/N_SENT): {0} {1} {2}'.format(total_loss/total_seen, total_loss, total_seen))
    return costs

def validate_helper(config, sess):
    model, saver = create_model(config, sess)
    valid_text_iterator = TextIterator(
                        source=config.valid_source_dataset,
                        target=config.valid_target_dataset,
                        source_dicts=config.source_dicts,
                        target_dict=config.target_dict,
                        batch_size=config.valid_batch_size,
                        maxlen=config.maxlen,
                        source_vocab_sizes=config.source_vocab_sizes,
                        target_vocab_size=config.target_vocab_size,
                        shuffle_each_epoch=False,
                        sort_by_length=False, #TODO
                        use_factor=(config.factors > 1),
                        maxibatch_size=config.maxibatch_size)
    costs = validate(config, sess, valid_text_iterator, model)
    lines = open(config.valid_target_dataset).readlines()
    for cost, line in zip(costs, lines):
        logging.info("{0} {1}".format(cost,line.strip()))



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
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs="+",
                         help="network vocabularies (one per source factor, plus target vocabulary)")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--model', '--saveto', type=str, default='model', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--reload', type=str, default=None, metavar='PATH',
                         help="load existing model from this path. Set to \"latest_checkpoint\" to reload the latest checkpoint in the same directory of --saveto")
    data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                         help="don't reload training progress (only used if --reload is enabled)")
    data.add_argument('--summary_dir', type=str, required=False, metavar='PATH', 
                         help="directory for saving summaries (default: same directory as the --saveto file)")
    data.add_argument('--summaryFreq', type=int, default=0, metavar='INT',
                         help="Save summaries after INT updates, if 0 do not save summaries (default: %(default)s)")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--embedding_size', '--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--state_size', '--dim', type=int, default=1000, metavar='INT',
                         help="hidden state size (default: %(default)s)")

    network.add_argument('--source_vocab_sizes', '--n_words_src', type=int, default=None, nargs='+', metavar='INT',
                         help="source vocabulary sizes (one per input factor) (default: %(default)s)")

    network.add_argument('--target_vocab_size', '--n_words', type=int, default=-1, metavar='INT',
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
    network.add_argument('--dropout_source', type=float, default=0, metavar="FLOAT",
                         help="dropout source words (0: no dropout) (default: %(default)s)")
    network.add_argument('--dropout_target', type=float, default=0, metavar="FLOAT",
                         help="dropout target words (0: no dropout) (default: %(default)s)")
    network.add_argument('--use_layer_norm', '--layer_normalisation', action="store_true", dest="use_layer_norm",
                         help="Set to use layer normalization in encoder and decoder")
    network.add_argument('--tie_decoder_embeddings', action="store_true", dest="tie_decoder_embeddings",
                         help="tie the input embeddings of the decoder with the softmax output embeddings")
    network.add_argument('--output_hidden_activation', type=str, default='tanh',
                         choices=['tanh', 'relu', 'prelu', 'linear'],
                         help='activation function in hidden layer of the output network (default: %(default)s)')

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
    training.add_argument('--decay_c', type=float, default=0, metavar='FLOAT',
                         help="L2 regularization penalty (default: %(default)s)")
    training.add_argument('--map_decay_c', type=float, default=0, metavar='FLOAT',
                         help="MAP-L2 regularization penalty towards original weights (default: %(default)s)")
    training.add_argument('--prior_model', type=str, metavar='PATH',
                         help="Prior model for MAP-L2 regularization. Unless using \"--reload\", this will also be used for initialization.")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--learning_rate', '--lrate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--keep_train_set_in_memory', action="store_true", 
                         help="Keep training dataset lines stores in RAM during training")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument('--optimizer', type=str, default="adam",
                         choices=['adam'],
                         help="optimizer (default: %(default)s)")

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
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--run_validation', action='store_true',
                         help="Compute validation score on validation dataset")

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
    translate.add_argument('--translate_valid', action='store_true', dest='translate_valid',
                            help='Translate source dataset instead of training')
    translate.add_argument('--no_normalize', action='store_false', dest='normalize',
                            help="Cost of sentences will not be normalized by length")
    translate.add_argument('--n_best', action='store_true', dest='n_best',
                            help="Print full beam")
    translate.add_argument('--n_threads', type=int, default=5, metavar='INT',
                         help="Number of threads to use for beam search (default: %(default)s)")
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

    if len(config.dictionaries) != config.factors + 1:
        logging.error('\'--dictionaries\' must specify one dictionary per source factor and one target dictionary\n')
        sys.exit(1)

    # set vocabulary sizes
    vocab_sizes = []
    if config.source_vocab_sizes == None:
        vocab_sizes = [-1] * config.factors
    elif len(config.source_vocab_sizes) == config.factors:
        vocab_sizes = config.source_vocab_sizes
    elif len(config.source_vocab_sizes) < config.factors:
        num_missing = config.factors - len(config.source_vocab_sizes)
        vocab_sizes += config.source_vocab_sizes + [-1] * num_missing
    else:
        logging.error('too many values supplied to \'--source_vocab_sizes\' option (expected one per factor = {0})'.format(config.factors))
        sys.exit(1)
    if config.target_vocab_size == -1:
        vocab_sizes.append(-1)
    else:
        vocab_sizes.append(config.target_vocab_size)

    # for unspecified vocabulary sizes, determine sizes from vocabulary dictionaries
    for i, vocab_size in enumerate(vocab_sizes):
        if vocab_size >= 0:
            continue
        try:
            d = load_dict(config.dictionaries[i])
        except Exception as e:
            print(e)
            logging.error('failed to determine vocabulary size from file: {0}'.format(config.dictionaries[i]))
        vocab_sizes[i] = max(d.values()) + 1

    config.source_dicts = config.dictionaries[:-1]
    config.source_vocab_sizes = vocab_sizes[:-1]
    config.target_dict = config.dictionaries[-1]
    config.target_vocab_size = vocab_sizes[-1]


    # set the model version
    config.model_version = 0.2

    return config

if __name__ == "__main__":

    # set up logging
    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    config = parse_args()
    logging.info(config)
    with tf.Session() as sess:
        if config.translate_valid:
            translate(config, sess)
        elif config.run_validation:
            validate_helper(config, sess)
        else:
            train(config, sess)
