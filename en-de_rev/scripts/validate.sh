#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the newstest2013 test set
# using detokenized BLEU (equivalent to evaluation with mteval-v13a.pl).

translations=$1

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/scripts/
main_dir=$script_dir/../
# data_dir=$main_dir/data
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/


#language-independent variables (toolkit locations)
. $main_dir/vars

#language-dependent variables (source and target language)
. $main_dir/vars

dev_prefix=newstest2013
ref=$data_dir/$dev_prefix.$trg

# write resulting file
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
$script_dir/postprocess.sh < $translations > /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/output/out_$dev_prefix_$current_time.$trg
 
# evaluate translations and write BLEU score to standard output (for
# use by nmt.py)
$script_dir/postprocess.sh < $translations | \
    $nematus_home/data/multi-bleu-detok.perl $ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
