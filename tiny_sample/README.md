Originally from:

  https://github.com/rsennrich/wmt16-scripts/tree/master/factored_sample

I've run the preprocessing through moses and checked in *tiny*
versions of the output.  The goal is to have a working version of
nematus to run locally (e.g. on Mac, without GPU), that will exercise
whether nematus can go through a full train run without crashing.

Factors have been removed for simplicity.

```
$ wc -l data/*
   100 data/100_corpus.bpe.de
   100 data/100_corpus.bpe.en
    30 data/30_newstest2013.bpe.de
    30 data/30_newstest2013.bpe.en
 16827 data/corpus.bpe.de.json
 10568 data/corpus.bpe.en.json
```

Setup your virtualenv (once):

```
[mbp-isi-joelb ~/views/flyover/nematus (joelb-initial *+)]
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

Run nematus training:

```
$ cd tiny_sample
$ PYTHONPATH=.. PYTHONUNBUFFERED=1 \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu,on_unused_input=warn \
time python small_config.py
```
