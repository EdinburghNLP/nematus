#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

mkdir -p models

../nematus/nmt.py \
  --model models/model_drt_4.npz \
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
  --dispFreq 500 \
  --finish_after 500 \
  --dec_base_recurrence_transition_depth 4
