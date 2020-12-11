#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

../nematus/train.py \
  --model models/model.npz \
  --datasets data/corpus.en data/corpus.de \
  --dictionaries data/vocab.json data/vocab.json \
  --n_words_src 10000 \
  --n_words 10000 \
  --model_type transformer \
  --embedding_size 128 \
  --tie_encoder_decoder_embeddings \
  --tie_decoder_embeddings \
  --state_size 128 \
  --transformer_enc_depth 2 \
  --transformer_dec_depth 2 \
  --transformer_ffn_hidden_size 256 \
  --loss_function per-token-cross-entropy \
  --clip_c 0.0 \
  --label_smoothing 0.1 \
  --optimizer adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-09 \
  --learning_schedule transformer \
  --warmup_steps 4000 \
  --maxlen 100 \
  --batch_size 300 \
  --token_batch_size 3000 \
  --disp_freq 500 \
  --finish_after 500
