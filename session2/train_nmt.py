import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
<<<<<<< HEAD
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=1000,
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=100,
                     sampleFreq=100,
                     datasets=['../data/hal/train/tok/en', 
                               '../data/hal/train/tok/ja'],
                     valid_datasets=['../data/hal/dev/tok/en',
                                     '../data/hal/dev/tok/ja'],
                     dictionaries=['../data/hal/train/tok/en.pkl', 
                                   '../data/hal/train/tok/ja.pkl'],
                     use_dropout=params['use-dropout'][0])
=======
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        clip_c=params['clip-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=50,
                                        batch_size=32,
                                        valid_batch_size=32,
					datasets=['/ichec/home/users/%s/data/europarl-v7.fr-en.en.tok'%os.environ['USER'], 
					'/ichec/home/users/%s/data/europarl-v7.fr-en.fr.tok'%os.environ['USER']],
					valid_datasets=['/ichec/home/users/%s/data/newstest2011.en.tok'%os.environ['USER'], 
					'/ichec/home/users/%s/data/newstest2011.fr.tok'%os.environ['USER']],
					dictionaries=['/ichec/home/users/%s/data/europarl-v7.fr-en.en.tok.pkl'%os.environ['USER'], 
					'/ichec/home/users/%s/data/europarl-v7.fr-en.fr.tok.pkl'%os.environ['USER']],
                                        validFreq=5000,
                                        dispFreq=10,
                                        saveFreq=5000,
                                        sampleFreq=1000,
                                        use_dropout=params['use-dropout'][0])
>>>>>>> 9738dff8f1f2018f4ffbdc11c4de27763fcb168e
    return validerr

if __name__ == '__main__':
    main(0, {
<<<<<<< HEAD
        'model': ['model_hal.npz'],
        'dim_word': [512],
=======
        'model': ['/ichec/home/users/%s/models/model_session2.npz'%os.environ['USER']],
        'dim_word': [500],
>>>>>>> 9738dff8f1f2018f4ffbdc11c4de27763fcb168e
        'dim': [1024],
        'n-words': [30000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'clip-c': [1.], 
<<<<<<< HEAD
        'use-dropout': [True],
=======
        'use-dropout': [False],
>>>>>>> 9738dff8f1f2018f4ffbdc11c4de27763fcb168e
        'learning-rate': [0.0001],
        'reload': [True]})


