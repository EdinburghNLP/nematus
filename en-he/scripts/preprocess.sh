#!/bin/sh
# Distributed under MIT license

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/scripts/
main_dir=$script_dir/..
data_dir=$main_dir/data
model_dir=$main_dir/model

# variables (toolkits; source and target language)
. $main_dir/vars

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=40000

#minimum number of times we need to have seen a character sequence in the training text before we merge it into one unit
#this is applied to each training text independently, even with joint BPE
bpe_threshold=50

# tokenize
for prefix in corpus newstest2013 newstest2014 newstest2015 newstest2016 newstest2017
 do
   cat $data_dir/$prefix.$src | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $data_dir/$prefix.tok.$src

   cat $data_dir/$prefix.$trg | \
   $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
   $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $data_dir/$prefix.tok.$trg

 done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$moses_scripts/training/clean-corpus-n.perl $data_dir/corpus.tok $src $trg $data_dir/corpus.tok.clean 1 80

# train truecaser
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$src -model $model_dir/truecase-model.$src
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$trg -model $model_dir/truecase-model.$trg

# apply truecaser (cleaned training corpus)
for prefix in corpus
 do
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/$prefix.tok.clean.$src > $data_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $data_dir/$prefix.tok.clean.$trg > $data_dir/$prefix.tc.$trg
 done

# apply truecaser (dev/test files)
for prefix in newstest2013 newstest2014 newstest2015 newstest2016 newstest2017
 do
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/$prefix.tok.$src > $data_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $data_dir/$prefix.tok.$trg > $data_dir/$prefix.tc.$trg
 done

# train BPE
$bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/corpus.tc.$src $data_dir/corpus.tc.$trg --write-vocabulary $data_dir/vocab.$src $data_dir/vocab.$trg -s $bpe_operations -o $model_dir/$src$trg.bpe

# apply BPE

for prefix in corpus newstest2013 newstest2014 newstest2015 newstest2016 newstest2017
 do
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.tc.$src > $data_dir/$prefix.bpe.$src
  $bpe_scripts/apply_bpe.py -c $model_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.tc.$trg > $data_dir/$prefix.bpe.$trg
 done

# build network dictionaries for separate source / target vocabularies
$nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg

# build network dictionary for combined source + target vocabulary (for use
# with tied encoder-decoder embeddings)
cat $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg > $data_dir/corpus.bpe.both
$nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.both
