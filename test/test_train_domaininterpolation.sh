#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

mkdir -p models

../nematus/nmt.py \
  --model models/model_domainadapt.npz \
  --datasets data/corpus.en data/corpus.de \
  --dictionaries data/vocab.en.json data/vocab.de.json \
  --dim_word 256 \
  --dim 512 \
  --n_words_src 30000 \
  --n_words 30000 \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --no_shuffle \
  --dispFreq 100 \
  --finish_after 50000 \
  --domain_interpolation_indomain_datasets data/indomain-corpus.en data/indomain-corpus.de \
  --domain_interpolation_min 0.5 \
  --domain_interpolation_max 1.0 \
  --domain_interpolation_inc 0.2 \
  --saveFreq 100 \
  --valid_datasets data/indomain-dev.en data/indomain-dev.de \
  --valid_batch_size 20 \
  --validFreq 100 \
  --patience 3 \
  --use_domain_interpolation \
#  --reload

