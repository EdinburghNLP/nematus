import os

from lm import train


def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=30,
        batch_size=32,
        valid_batch_size=16,
        validFreq=5000,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset='/ichec/work/dl4mt_data/nec_files/wiki.tok.txt.gz',
        valid_dataset='/ichec/work/dl4mt_data/nec_files/newstest2011.en.tok',
        dictionary='/ichec/work/dl4mt_data/nec_files/wiki.tok.txt.gz.pkl',
        use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/ichec/home/users/%s/models/model_session0.npz' %
                  os.environ['USER']],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
