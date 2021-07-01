#!/bin/bash

translations=$1
processed=$2

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/scripts/
main_dir=$script_dir/../
# data_dir=$main_dir/data
# data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
tmp="tmp_postprocessed$current_time"
$script_dir/postprocess.sh < $translations > "$tmp"

python $remove_edges "$tmp" -o $processed
rm $tmp