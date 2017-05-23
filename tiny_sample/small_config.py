import numpy
import os
import sys

VOCAB_SIZE = 90000
SRC = "en"
TGT = "de"
DATA_DIR = "data/"

from nematus.nmt import train

# joelb: remove the factors, just run simple
if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                     reload_=False,      # joelb: for now, overwrite my small model
                    dim_word=500,
                    dim=1024,
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=50,
                    batch_size=45,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/100_corpus.bpe.' + SRC, DATA_DIR + '/100_corpus.bpe.' + TGT],
                    valid_datasets=[DATA_DIR + '/30_newstest2013.bpe.' + SRC, DATA_DIR + '/30_newstest2013.bpe.' + TGT],
                    dictionaries=[DATA_DIR + '/corpus.bpe.' + SRC + '.json',
                                  DATA_DIR + '/corpus.bpe.' + TGT + '.json'],
                    validFreq=10000,
                    dispFreq=1,
                    saveFreq=10,
                    sampleFreq=10,
                    max_epochs=10,       # joelb: small so we never validate
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='validate.sh')
    print validerr
