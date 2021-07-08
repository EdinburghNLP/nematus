#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the newstest2013 test set
# using detokenized BLEU (equivalent to evaluation with mteval-v13a.pl).

translations=$1

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/scripts/
main_dir=$script_dir/../
# data_dir=$main_dir/data


#language-independent variables (toolkit locations)
. $main_dir/vars

#language-dependent variables (source and target language)
. $script_dir/vars

dev_prefix=dev
ref=$data_dir/$dev_prefix.$trg

# evaluate translations and write BLEU score to standard output (for
# use by nmt.py)
$script_dir/postprocess.sh < $translations | \
    $nematus_home/data/multi-bleu-detok.perl $ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
