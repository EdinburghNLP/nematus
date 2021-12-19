#!/bin/bash

model_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/scripts/

#language-independent variables (toolkit locations)
. $model_dir/../vars

#language-dependent variables (source and target language)
. $model_dir/vars

$model_dir/preprocess.sh | \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device python $nematus_home/nematus/translate.py \
     -m $model_dir/model.l2r.ens{1,2,3,4}.npz \
     -k 12 -n -p 1 --suppress-unk | \
$model_dir/postprocess.sh
