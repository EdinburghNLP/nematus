#!/bin/bash

model_dir=`dirname $0`
model_dir=/cs/snapless/oabend/borgr/TG/en-de_rev/scripts/

#language-independent variables (toolkit locations)
. $model_dir/../vars

#language-dependent variables (source and target language)
. $model_dir/vars

sed 's/\@\@ //g' | \
$moses_scripts/recaser/detruecase.perl | \
$moses_scripts/tokenizer/detokenizer.perl -l $trg