import tensorflow as tf
from tf_layers import *
from data_iterator import TextIterator
import time
import argparse
from tf_model import *
from util import *
import os
from threading import Thread
from Queue import Queue
from datetime import datetime

def create_model(config, sess):
    print >>sys.stderr, 'Building model...',
    model = StandardModel(config)

    # initialize model
    saver = tf.train.Saver(max_to_keep=None)
    if not config.reload:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    else:
        saver.restore(sess, os.path.abspath(config.reload))
    print >>sys.stderr, 'Done'

    return model, saver 

def load_data(config):
    print >>sys.stderr, 'Reading data...',
    text_iterator = TextIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_dicts=[config.source_vocab],
                        target_dict=config.target_vocab,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        n_words_source=config.source_vocab_size,
                        n_words_target=config.target_vocab_size,
                        skip_empty=True,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        maxibatch_size=config.maxibatch_size)

    if config.validFreq:
        valid_text_iterator = TextIterator(
                            source=config.valid_source_dataset,
                            target=config.valid_target_dataset,
                            source_dicts=[config.source_vocab],
                            target_dict=config.target_vocab,
                            batch_size=config.valid_batch_size,
                            maxlen=config.validation_maxlen,
                            n_words_source=config.source_vocab_size,
                            n_words_target=config.target_vocab_size,
                            shuffle_each_epoch=False,
                            sort_by_length=True,
                            maxibatch_size=config.maxibatch_size)
    else:
        valid_text_iterator = None
    print >>sys.stderr, 'Done'
    return text_iterator, valid_text_iterator

def load_dictionaries(config):
    source_to_num = load_dict(config.source_vocab)
    target_to_num = load_dict(config.target_vocab)
    num_to_source = reverse_dict(source_to_num)
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target

def read_all_lines(config, path):
    source_to_num, _, _, _ = load_dictionaries(config)
    lines = map(lambda l: l.strip().split(), open(path, 'r').readlines())
    fn = lambda w: [source_to_num[w] if w in source_to_num else 1] # extra [ ] brackets for factor dimension
    lines = map(lambda l: map(lambda w: fn(w), l), lines)
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
    model, saver = create_model(config, sess)

    x,x_mask,y,y_mask = model.get_score_inputs()
    apply_grads = model.get_apply_grads()
    t = model.get_global_step()
    loss_per_sentence = model.get_loss()
    mean_loss = model.get_mean_loss()

    if config.summaryFreq:
        writer = tf.summary.FileWriter(config.summary_dir, sess.graph)
    tf.summary.scalar(name='mean cost', tensor=mean_loss)
    tf.summary.scalar(name='t', tensor=t)
    merged = tf.summary.merge_all()

    text_iterator, valid_text_iterator = load_data(config)
    source_to_num, target_to_num, num_to_source, num_to_target = load_dictionaries(config)
    total_loss = 0.
    n_sents, n_words = 0, 0
    last_time = time.time()
    uidx = sess.run(t)
    print >>sys.stderr, "Initial uidx={}".format(uidx)
    STOP = False
    for eidx in xrange(config.max_epochs):
        print 'Starting epoch', eidx
        for source_sents, target_sents in text_iterator:
            x_in, x_mask_in, y_in, y_mask_in = prepare_data(source_sents, target_sents, maxlen=config.maxlen)
            if x_in is None:
                print >>sys.stderr, 'Minibatch with zero sample under length ', config.maxlen
                continue
            (seqLen, batch_size) = x_in.shape
            inn = {x:x_in, y:y_in, x_mask:x_mask_in, y_mask:y_mask_in}
            out = [t, apply_grads, mean_loss]
            if config.summaryFreq and uidx % config.summaryFreq == 0:
                out += [merged]
            out = sess.run(out, feed_dict=inn)
            mean_loss_out = out[2]
            total_loss += mean_loss_out*batch_size
            n_sents += batch_size
            n_words += int(numpy.sum(y_mask_in))
            uidx += 1

            if config.summaryFreq and uidx % config.summaryFreq == 0:
                writer.add_summary(out[3], out[0])

            if config.dispFreq and uidx % config.dispFreq == 0:
                duration = time.time() - last_time
                disp_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                print disp_time, \
                      'Epoch:', eidx, \
                      'Update:', uidx, \
                      'Loss/word:', total_loss/n_words, \
                      'Words/sec:', n_words/duration, \
                      'Sents/sec:', n_sents/duration
                last_time = time.time()
                total_loss = 0.
                n_sents = 0
                n_words = 0

            if config.saveFreq and uidx % config.saveFreq == 0:
                saver.save(sess, save_path=config.saveto, global_step=uidx)

            if config.sampleFreq and uidx % config.sampleFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :10], x_mask_in[:, :10], y_in[:, :10]
                samples = model.sample(sess, x_small, x_mask_small)
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    print >>sys.stderr, 'SOURCE:', seqs2words(xx, num_to_source)
                    print >>sys.stderr, 'TARGET:', seqs2words(yy, num_to_target)
                    print >>sys.stderr, 'SAMPLE:', seqs2words(ss, num_to_target)

            if config.beamFreq and uidx % config.beamFreq == 0:
                x_small, x_mask_small, y_small = x_in[:, :10], x_mask_in[:, :10], y_in[:,:10]
                samples = model.beam_search(sess, x_small, x_mask_small, config.beam_size)
                # samples is a list with shape batch x beam x len
                assert len(samples) == len(x_small.T) == len(y_small.T), (len(samples), x_small.shape, y_small.shape)
                for xx, yy, ss in zip(x_small.T, y_small.T, samples):
                    print >>sys.stderr, 'SOURCE:', seqs2words(xx, num_to_source)
                    print >>sys.stderr, 'TARGET:', seqs2words(yy, num_to_target)
                    for i, (sample, cost) in enumerate(ss):
                        print >>sys.stderr, 'SAMPLE', i, ':', seqs2words(sample, num_to_target), 'Cost/Len/Avg:', cost, '/', len(sample), '/', cost/len(sample)

            if config.validFreq and uidx % config.validFreq == 0:
                validate(sess, valid_text_iterator, model)

            if config.finish_after and uidx % config.finish_after == 0:
                print >>sys.stderr, "Maximum number of updates reached"
                STOP=True
                break
        if STOP:
            break

def translate(config, sess):
    model, saver = create_model(config, sess)
    start_time = time.time()
    _, _, _, num_to_target = load_dictionaries(config)
    print >>sys.stderr, "NOTE: Length of translations is capped to {}".format(config.translation_maxlen)

    n_sent = 0
    batches, idxs = read_all_lines(config, config.valid_source_dataset)
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
        print >>sys.stderr, 'Translated {} sents'.format(n_sent)
    for _ in range(config.n_threads):
        in_queue.put(None)
    outputs = [beam for batch in outputs for beam in batch]
    outputs = numpy.array(outputs, dtype=numpy.object)
    outputs = outputs[idxs.argsort()]

    for beam in outputs:
        if config.normalize:
            beam = map(lambda (sent, cost): (sent, cost/len(sent)), beam)
        beam = sorted(beam, key=lambda (sent, cost): cost)
        if config.n_best:
            for sent, cost in beam:
                print seqs2words(sent, num_to_target), '[%f]' % cost
        else:
            best_hypo, cost = beam[0]
            print seqs2words(best_hypo, num_to_target)
    duration = time.time() - start_time
    print >> sys.stderr, 'Translated {} sents in {} sec. Speed {} sents/sec'.format(n_sent, duration, n_sent/duration)


def validate(sess, valid_text_iterator, model):
    costs = []
    total_loss = 0.
    total_seen = 0
    x,x_mask,y,y_mask = model.get_score_inputs()
    loss_per_sentence = model.get_loss()
    for x_v, y_v in valid_text_iterator:
        x_v_in, x_v_mask_in, y_v_in, y_v_mask_in = prepare_data(x_v, y_v, maxlen=None)
        feeds = {x:x_v_in, x_mask:x_v_mask_in, y:y_v_in, y_mask:y_v_mask_in}
        loss_per_sentence_out = sess.run(loss_per_sentence, feed_dict=feeds)
        total_loss += loss_per_sentence_out.sum()
        total_seen += x_v_in.shape[1]
        costs += list(loss_per_sentence_out)
        print >>sys.stderr, "Seen", total_seen
    print 'Validation loss (AVG/SUM/N_SENT):', total_loss/total_seen, total_loss, total_seen
    return costs

def validate_helper(config, sess):
    model, saver = create_model(config, sess)
    valid_text_iterator = TextIterator(
                        source=config.valid_source_dataset,
                        target=config.valid_target_dataset,
                        source_dicts=[config.source_vocab],
                        target_dict=config.target_vocab,
                        batch_size=config.valid_batch_size,
                        maxlen=config.validation_maxlen,
                        n_words_source=config.source_vocab_size,
                        n_words_target=config.target_vocab_size,
                        shuffle_each_epoch=False,
                        sort_by_length=False, #TODO
                        maxibatch_size=config.maxibatch_size)
    costs = validate(sess, valid_text_iterator, model)
    lines = open(config.valid_target_dataset).readlines()
    for cost, line in zip(costs, lines):
        print cost, line.strip()



def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--source_dataset', type=str, required=True, metavar='PATH', 
                         help="parallel training corpus (source)")
    data.add_argument('--target_dataset', type=str, required=True, metavar='PATH', 
                         help="parallel training corpus (target)")
    data.add_argument('--source_vocab', type=str, required=True, metavar='PATH', 
                         help="dictionary for the source data")
    data.add_argument('--target_vocab', type=str, required=True, metavar='PATH',
                         help="dictionary for the target data")
    data.add_argument('--saveFreq', type=int, default=30000, metavar='INT',
                         help="save frequency (default: %(default)s)")
    data.add_argument('--saveto', type=str, default='model', metavar='PATH', dest='saveto',
                         help="model file name (default: %(default)s)")
    data.add_argument('--reload', type=str, default=None, metavar='PATH',
                         help="load existing model from this path")
    data.add_argument('--summary_dir', type=str, required=False, metavar='PATH', 
                         help="directory for saving summaries")
    data.add_argument('--summaryFreq', type=int, default=0, metavar='INT',
                         help="Save summaries after INT updates (default: %(default)s)")

    network = parser.add_argument_group('network parameters')
    network.add_argument('--embedding_size', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--state_size', type=int, default=1000, metavar='INT',
                         help="hidden state size (default: %(default)s)")
    network.add_argument('--source_vocab_size', type=int, required=True, metavar='INT',
                         help="source vocabulary size (default: %(default)s)")
    network.add_argument('--target_vocab_size', type=int, required=True, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")
    network.add_argument('--nematus_compat', action='store_true',
                         help="Add this flag to have the same model architecture as Nematus(default: %(default)s)")


    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=50, metavar='INT',
                         help="maximum sequence length for training (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--finish_after', type=int, default=10000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--learning_rate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate (default: %(default)s)")
    training.add_argument('--no_shuffle', action="store_false", dest="shuffle_each_epoch",
                         help="disable shuffling of training data (for each epoch)")
    training.add_argument('--no_sort_by_length', action="store_false", dest="sort_by_length",
                         help='do not sort sentences in maxibatch by length')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                         help='size of maxibatch (number of minibatches that are sorted by length) (default: %(default)s)')
    training.add_argument('--use_layer_norm', action="store_true", dest="use_layer_norm",
                         help="Set to use layer normalization in encoder and decoder")

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_source_dataset', type=str, default=None, metavar='PATH', 
                         help="source validation corpus (default: %(default)s)")
    validation.add_argument('--valid_target_dataset', type=str, default=None, metavar='PATH',
                         help="target validation corpus (default: %(default)s)")
    validation.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validation minibatch size (default: %(default)s)")
    validation.add_argument('--validFreq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--run_validation', action='store_true',
                         help="Compute validation score on validation dataset")
    validation.add_argument('--validation_maxlen', type=int, default=999999, metavar='INT',
                         help="Sequences longer than this will not be used for validation (default: %(default)s)")

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
    return config

if __name__ == "__main__":
    config = parse_args()
    print >>sys.stderr, config
    with tf.Session() as sess:
        if config.translate_valid:
            translate(config, sess)
        elif config.run_validation:
            validate_helper(config, sess)
        else:
            train(config, sess)
