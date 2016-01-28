import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    basedir = '/data/lisatmp3/firatorh/nmt/europarlv7'
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=15,
                                        batch_size=32,
                                        valid_batch_size=32,
					datasets=['%s/europarl-v7.fr-en.fr.tok'%basedir,
					'%s/europarl-v7.fr-en.en.tok'%basedir],
					valid_datasets=['%s/newstest2011.fr.tok'%basedir,
					'%s/newstest2011.en.tok'%basedir],
					dictionaries=['%s/europarl-v7.fr-en.fr.tok.pkl'%basedir,
					'%s/europarl-v7.fr-en.en.tok.pkl'%basedir],
                                        validFreq=500000,
                                        dispFreq=1,
                                        saveFreq=100,
                                        sampleFreq=50,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False)
    return validerr

if __name__ == '__main__':
    basedir = '/data/lisatmp3/firatorh/nmt/europarlv7'
    main(0, {
        'model': ['%s/models/model_session3.npz'%basedir],
        'dim_word': [150],
        'dim': [124],
        'n-words': [3000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})


