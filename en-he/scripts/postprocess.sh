#!/bin/sh
# Distributed under MIT license

# this sample script postprocesses the MT output,
# including merging of BPE subword units,
# detruecasing, and detokenization

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/scripts/
main_dir=$script_dir/../

# variables (toolkits; source and target language)
. $main_dir/vars

sed -r 's/\@\@ //g' |
$moses_scripts/recaser/detruecase.perl |
$moses_scripts/tokenizer/detokenizer.perl -l $trg
