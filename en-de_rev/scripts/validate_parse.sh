#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the newstest2013 test set
# using detokenized BLEU (equivalent to evaluation with mteval-v13a.pl).

translations=$1

script_dir=`dirname $0`
script_dir=/cs/snapless/oabend/borgr/TG/en-de/scripts/
main_dir=$script_dir/../
# data_dir=$main_dir/data
# data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD

#language-independent variables (toolkit locations)
. $main_dir/vars

#language-dependent variables (source and target language)
. $main_dir/vars
postprocess_to_trans="${nematus_home}/nematus/parsing/postprocess_to_trans.py"

dev_prefix=newstest2013
ref=$data_dir/$dev_prefix.parse.$trg

# create ref file if needed
if [ ! -f $ref ] ; then
	trans=$data_dir/$dev_prefix.trans.$trg
	if [ ! -f $trans ] ; then
		$script_dir/postprocess.sh < "$data_dir/newstest2013.unesc.tok.tc.bpe.trns.$trg" > "$ref"
	fi
	python $postprocess_to_trans $trans -o $ref
fi

# write resulting file
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
$script_dir/postprocess.sh < "$translations" > "/cs/snapless/oabend/borgr/TG/en-de/output/out_$dev_prefix_$current_time.$trg"
 
# evaluate translations and write BLEU score to standard output (for
# use by nmt.py)
tmp="tmp_postprocessed$current_time"
$script_dir/postprocess.sh < $translations > "$tmp.out" 
python postprocess_to_trans "$tmp.out" |\
    $nematus_home/data/multi-bleu-detok.perl $ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
rm "$tmp.out"