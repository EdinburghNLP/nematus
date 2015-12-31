#!/bin/bash

# source language (example: fr)
S=$1
# target language (example: en)
T=$2

# path to dl4mt/data
P1=$3

# path to subword NMT scripts (can be downloaded from https://github.com/rsennrich/subword-nmt)
P2=$4

# merge all parallel corpora
./merge.sh $1 $2

# tokenize
perl $P1/tokenizer.perl -threads 5 -l $S < all_${S}-${T}.${S} > all_${S}-${T}.${S}.tok
perl $P1/tokenizer.perl -threads 5 -l $T < all_${S}-${T}.${T} > all_${S}-${T}.${T}.tok

# BPE
if [ ! -f "${S}.bpe" ]; then
    python $P2/learn_bpe.py -s 20000 < all_${S}-${T}.${S}.tok > ${S}.bpe
fi
if [ ! -f "${T}.bpe" ]; then
    python $P2/learn_bpe.py -s 20000 < all_${S}-${T}.${T}.tok > ${T}.bpe
fi

python $P2/apply_bpe.py -c ${S}.bpe < all_${S}-${T}.${S}.tok > all_${S}-${T}.${S}.tok.bpe
python $P2/apply_bpe.py -c ${T}.bpe < all_${S}-${T}.${T}.tok > all_${S}-${T}.${T}.tok.bpe

# shuffle 
python $3/shuffle.py all_${S}-${T}.${S}.tok.bpe all_${S}-${T}.${T}.tok.bpe

# build dictionary
python $P1/build_dictionary.py all_${S}-${T}.${S}.tok.bpe
python $P1/build_dictionary.py all_${S}-${T}.${T}.tok.bpe

