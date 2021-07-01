#!/bin/bash

model_dir=`dirname $0`
script_dir=/cs/snapless/oabend/borgr/TG/en-de/scripts/

#language-independent variables (toolkit locations)
. $model_dir/../vars

#language-dependent variables (source and target language)
. $model_dir/vars

$model_dir/preprocess.sh | \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device python $nematus_home/nematus/translate.py \
     -m $model_dir/model.l2r.ens1.npz \
     -k 12 -n -p 1 --suppress-unk | \
$model_dir/postprocess.sh
