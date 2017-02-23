import tensorflow as tf
from tf_layers import *
from data_iterator import TextIterator
import time
import argparse
from tf_model import *




# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    assert n_factors == 1
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    # there is only one factor, get rid of that dimension
    x = x.squeeze(axis=0)

    return x, x_mask, y, y_mask



def test_forward_step(config):
    x, x_mask, y, y_mask, logits = build_model(config)
    batch = config.batch_size
    x_seqLen=13
    y_seqLen=17
    x_in = numpy.random.randint(config.source_vocab_size, size=(x_seqLen,batch))
    y_in = numpy.random.randint(config.target_vocab_size, size=(y_seqLen,batch))
    x_lens = numpy.random.randint(1, x_seqLen,size=batch)
    y_lens = numpy.random.randint(1, y_seqLen,size=batch)
    x_mask_in = numpy.zeros((x_seqLen, batch))
    y_mask_in = numpy.zeros((y_seqLen, batch))

    for i in range(x_lens.shape[0]):
        x_mask_in[:x_lens[i], i] = 1.

    for i in range(y_lens.shape[0]):
        y_mask_in[:y_lens[i], i] = 1.0

    ins = {x:x_in, y:y_in, x_mask:x_mask_in, y_mask:y_mask_in}
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        ans = sess.run(logits, feed_dict=ins)
        print 'got', ans.shape
        print 'expected', (y_seqLen, batch, config.target_vocab_size)


#def prepare_dictionaries(config):
#    
#    def check_vocabulary(vocab, size):
#        assert range(size) == sorted(vocab.keys()), "Keys should be 0...(n-1)"
#        assert len(set(vocab.values())) == size, "Duplicate words!"
#
#    def invert_dictionary(vocab):
#        keys, values = zip(*vocab.items())
#        return dict(zip(values,keys))
#
#    source_vocab = json.load(config.source_vocab)
#    check_vocabulary(source_vocab, config.source_vocab_size)
#    source_vocab = invert_dictionary(source_vocab)
#
#    target_vocab = json.load(config.target_vocab)
#    check_vocabulary(target_vocab, target_vocab_size)
#    target_vocab = invert_dictionary(target_vocab)
#
#    return source_vocab, target_vocab

def train(config):
    print 'Building model'
    model = StandardModel(config)

    x,x_mask,y,y_mask = model.get_score_inputs()
    loss_per_sentence = model.get_loss()
    mean_loss = tf.reduce_mean(loss_per_sentence, keep_dims=False)
    print 'Getting samples'
    sampled_ys = model.get_samples()
    print 'Done'

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    t = tf.Variable(0, name='time', trainable=False, dtype=tf.int32)
    grad_vars = optimizer.compute_gradients(mean_loss)
    grads, varss = zip(*grad_vars)

    clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_c)
    # Might be interesting to see how the global norm changes over time
    grad_vars = zip(clipped_grads, varss)
    apply_grads = optimizer.apply_gradients(grad_vars, global_step=t)

    print 'Reading data...',
    text_iterator = TextIterator(
                        source=config.source_dataset,
                        target=config.target_dataset,
                        source_dicts=[config.source_vocab],
                        target_dict=config.target_vocab,
                        batch_size=config.batch_size,
                        maxlen=config.maxlen,
                        n_words_source=config.source_vocab_size,
                        n_words_target=config.target_vocab_size,
                        shuffle_each_epoch=config.shuffle_each_epoch,
                        sort_by_length=config.sort_by_length,
                        maxibatch_size=config.maxibatch_size)
    print 'Done'

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=10)
        if not config.reload:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
        else:
            saver.restore(sess, config.reload)

        n_samples = 0
        uidx = 0
        last_time = time.time()
        last_n_samples = n_samples
        for eidx in xrange(config.max_epochs):
            # get data and do the update
            #TODO: Add sampling, checkpointing, validation
            print 'Starting epoch', eidx
            for source_sents, target_sents in text_iterator:
                x_in, x_mask_in, y_in, y_mask_in = prepare_data(source_sents, target_sents, maxlen=config.maxlen)
                if x_in is None:
                    print 'Minibatch with zero sample under length ', config.maxlen
                    continue
                inn = {x:x_in, y:y_in, x_mask:x_mask_in, y_mask:y_mask_in}
                out = [t, apply_grads, mean_loss]
                t_out, _, mean_loss_out = sess.run(out, feed_dict=inn)
                uidx += 1
                n_samples += len(x_in)

                if uidx % config.dispFreq == 0:
                    print 'Epoch ', eidx, \
                           'Update ', uidx, \
                           'Cost ', mean_loss_out, \
                           'Sents/sec', (n_samples - last_n_samples) / (time.time() - last_time)
                    last_time = time.time()
                    last_n_samples = n_samples

                if uidx % config.saveFreq == 0:
                    saver.save(sess, save_path=config.saveto, global_step=uidx)

                if uidx % config.sampleFreq == 0:
                    print 'TIME TO SAMPLE'
                    samples = sess.run(sampled_ys, feed_dict=inn)
                    print 'Samples are:'
                    print samples


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

    network = parser.add_argument_group('network parameters')
    network.add_argument('--embedding_size', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--state_size', type=int, default=1000, metavar='INT',
                         help="hidden state size (default: %(default)s)")
    network.add_argument('--source_vocab_size', type=int, required=True, metavar='INT',
                         help="source vocabulary size (default: %(default)s)")
    network.add_argument('--target_vocab_size', type=int, required=True, metavar='INT',
                         help="target vocabulary size (default: %(default)s)")


    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=5000, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
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

    display = parser.add_argument_group('display parameters')
    display.add_argument('--dispFreq', type=int, default=1000, metavar='INT',
                         help="display loss after INT updates (default: %(default)s)")
    display.add_argument('--sampleFreq', type=int, default=10000, metavar='INT',
                         help="display some samples after INT updates (default: %(default)s)")


    config = parser.parse_args()
    return config

if __name__ == "__main__":
    config = parse_args()
    print config
    train(config)
    print 'Success'
