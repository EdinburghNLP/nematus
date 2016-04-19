import numpy
import os

import numpy
import os

from nematus import train

SRC = 'fr'
TRG = 'en'

if __name__ == '__main__':
    train(saveto='model.npz',
        reload_=True,
        dim_word=500,
        dim=1024,
        n_words_src=50000,
        n_words=50000,
        decay_c=0.,
        clip_c=1.,
        lrate=0.0001,
        optimizer='adadelta',
        maxlen=50,
        batch_size=50,
        valid_batch_size=50,
        datasets=['europarl-v7.' + SRC + '-' + TRG + '.' + SRC + '.tok.bpe',
                  'europarl-v7.' + SRC + '-' + TRG + '.' + TRG + '.tok.bpe'],
        valid_datasets=['newstest2011.' + SRC + 'tok.bpe',
                        'newstest2011.' + TRG + 'tok.bpe',
        dictionaries=['europarl-v7.' + SRC + '-' + TRG + '.' + SRC + '.tok.bpe.json',
                      'europarl-v7.' + SRC + '-' + TRG + '.' + TRG + '.tok.bpe.json'],
        validFreq=10000,
        dispFreq=1000,
        saveFreq=30000,
        sampleFreq=10000,
        use_dropout=False,
        dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
        dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
        dropout_source=0.1, # dropout source words (0: no dropout)
        dropout_target=0.1, # dropout target words (0: no dropout)
        overwrite=False,
        external_validation_script=WDIR + './validate.sh')
