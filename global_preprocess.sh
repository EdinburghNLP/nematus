#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/preprocess-%j.out

source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus_env3/bin/activate
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # fails on symlinks
#SCRIPT_FILE=`basename "$0"`
train_models=false
# suffix of source language files
SRC=en
# suffix of target language files (can be he, de, ru...)
TRG=$0
# preprocessing id
case ${TRG} in
	ru)
		ID=6.6
		;;
	de)
		ID=5.8
		;;
	he)
		ID=20.07.21
		;;
	*)
		echo "invalid language given. the possible languages are ru, de, he"
		;;
esac

 training lengths
min_length=3
max_length=100
ratio=1.5
alignment_score="-180"
cpus=4
fast_align=/cs/snapless/oabend/borgr/SSMT/fast_align/build/fast_align
filter_lines=/cs/snapless/oabend/borgr/SSMT/preprocess/filter_lines.py
#!#!#!# extras config #!#!#!#
# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=30000
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/cs/snapless/oabend/borgr/SSMT/preprocess/mosesdecoder
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/cs/snapless/oabend/borgr/SSMT/preprocess/subword-nmt
# path to unescaping scripts
unescape=/cs/snapless/oabend/borgr/SSMT/preprocess/unescape.py
# nematus dir
nematus_home=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/
datadir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/${SRC}-${TRG}
traindir=/cs/snapless/oabend/borgr/SSMT/data/${SRC}_${TRG}
workdir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/${SRC}-${TRG}/preprocess/
mkdir -p "$workdir"
modeldir=/cs/snapless/oabend/borgr/SSMT/preprocess/model/${ID}
mkdir -p "$modeldir"
## copy script file to dir
#cp "${SCRIPT_DIR}/${SCRIPT_FILE}" "${workdir}/preprocess.sh"
#echo "coppied ${SCRIPT_DIR}/${SCRIPT_FILE} to ${workdir}/preprocess.sh"
# corpora
train="train"
apply_to=('anti')

# choose if tc should be used
capitalized_langs=(en de fr es ru)
if [[ " ${capitalized_langs[@]} " =~ " ${TRG} " ]]; then
  trg_tc=".tc"
else
  trg_tc=""
fi
if [[ " ${capitalized_langs[@]} " =~ " ${SRC} " ]]; then
  src_tc=".tc"
else
  src_tc=""
fi

echo "################### start check lengths ###################"
# check lengths match in train
srclen=$(cat $traindir/$train.$SRC | wc -l)
trglen=$(cat $traindir/$train.$TRG | wc -l)
if [ $srclen != $trglen ] ; then
  echo "$traindir/$train.$SRC does not match the length of $traindir/$train.$TRG"
  exit 1
fi
# check lengths match in non-train corpora
for prefix in "${apply_to[@]}"
  do
  srclen=$(cat $datadir/$prefix.$SRC | wc -l)
  trglen=$(cat $datadir/$prefix.$TRG | wc -l)
  if [ $srclen != $trglen ] ; then
    echo "srclen: $srclen, srclen: $trglen"
    echo "$datadir/$prefix.$SRC does not match the length of $datadir/$prefix.$TRG"
    exit 1
  fi
done
echo "done checking lengths"

echo "################### start remove weird characters ###################"
# remove weird characters and insert = instead of - without spaces
if [ ! -f "$workdir/${train}.cln.$SRC" ]; then
    python /cs/snapless/oabend/borgr/SSMT/preprocess/preprocess.py -s $SRC -t $TRG -f $traindir -n "${train}" -o "${train}.cln" -d $workdir
fi
if [ ! -f "$workdir/$train.cln.$TRG" ]; then
    # remove lines dropped in target if --trg or -t not passed to preprocess.py
    python /cs/snapless/oabend/borgr/SSMT/preprocess/subsample_lines.py  -i $traindir/$train.$TRG -o $workdir/$train.cln.$TRG -d "${workdir}/${train}.idx"
fi
if [ ! -f "$workdir/$train.cln.$TRG" ]; then
    echo "failed to create $workdir/$train.cln.$TRG"
    exit 1
fi
# leave original copy at the base data dir
for prefix in "${apply_to[@]}"
 do
    if [ ! -f "$workdir/$prefix.$SRC" ]; then
        echo "$workdir/$prefix.$SRC"
        cp "$datadir/$prefix.$SRC" "$workdir/$prefix.$SRC"
    fi
    if [ ! -f "$workdir/$prefix.$TRG" ]; then
        echo "$workdir/$prefix.$TRG"
        cp "$datadir/$prefix.$TRG" "$workdir/$prefix.$TRG"
    fi
 done
echo "done remove weird characters and insert = instead of - without spaces"

srclen=$(cat "$workdir/${train}.cln.$SRC" | wc -l)
trglen=$(cat "$workdir/${train}.cln.$TRG" | wc -l)
if [ $srclen != $trglen ] ; then
  echo "$workdir/${train}.cln.$SRC does not match the length of $workdir/${train}.cln.$TRG"
  exit 1
fi

apply_to+=("${train}.cln")

echo "################### start tokenize ###################"
# tokenize
# also unescape ##AT-[...] to =, change html to text etc.
for prefix in "${apply_to[@]}"
 do
   echo "prefix $prefix"
    if [ ! -f $workdir/$prefix.unesc.tok.$SRC ] ; then
      echo $SRC $prefix
      python $unescape $workdir/$prefix.$SRC
      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.$SRC -out $workdir/$prefix.unesc.tok.$SRC -cmd "$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl $SRC < %s |
      $mosesdecoder/scripts/tokenizer/tokenizer_PTB.perl -l $SRC |
      $mosesdecoder/scripts/tokenizer/deescape-special-chars-PTB.perl > %s " -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.$SRC
    fi
  if [ ! -f $workdir/$prefix.unesc.tok.$SRC ] ; then
      echo "Failed in source tokenization"
      exit 1
  fi

    if [ ! -f $workdir/$prefix.unesc.tok.$TRG ] ; then
      echo $TRG $prefix
      python $unescape $workdir/$prefix.$TRG
      #cp $workdir/$prefix.unesc.$TRG $workdir/$prefix.unesc.tok.$TRG
       $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.$TRG -out $workdir/$prefix.unesc.tok.$TRG -cmd "$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl $TRG < %s |
       $mosesdecoder/scripts/tokenizer/tokenizer_PTB.perl -l $TRG |
       $mosesdecoder/scripts/tokenizer/deescape-special-chars-PTB.perl > %s " -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.$TRG
    fi
    if [ ! -f $workdir/$prefix.unesc.tok.$TRG ] ; then
      echo "$workdir/$prefix.unesc.tok.TRG"
      echo "Failed in target tokenization"
      exit 1
    fi
 done
echo "done tokenize"

if [ ! -f "$workdir/$train.cln.unesc.tok.$TRG" ]; then
    echo "$workdir/$train.cln.unesc.tok.$TRG not found, Something failed"
    exit 1
fi

echo "################### start clean ###################"
#clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
if [ ! -f "$workdir/$train.clean.unesc.tok.$SRC" ]; then
  echo "filtering lines"
#  $mosesdecoder/scripts/training/clean-corpus-n.perl $workdir/$train.cln.unesc.tok $SRC $TRG $workdir/$train.fltr.unesc.tok $min_length $max_length
  python $filter_lines $workdir/$train.clean.unesc.tok $SRC $TRG $workdir/$train.cln.unesc.tok.$SRC $workdir/$train.cln.unesc.tok.$TRG --ratio $ratio --min $min_length --max $max_length --fast_align $fast_align --fast_score $alignment_score
  echo "Consider the min alignment_score to use.\n probs.png - a plot to help in this choice \n Currently set to: ${alignment_score} use --force to recalculate or delete relevant files to only recalculate them"

fi

apply_to[-1]="${train}.clean"

echo "################### start truecaser ###################"
# train truecaser
if [ $train_models = true ]; then
  if [ ! -f "$modeldir/truecase-model.$train.$SRC" ]; then
    echo "train src truecase $modeldir/truecase-model.$train.$SRC"
    $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $workdir/$train.clean.unesc.tok.$SRC -model $modeldir/truecase-model.$train.$SRC

  fi
  if [ ! -f "$modeldir/truecase-model.$train.$TRG" ]; then
    echo "train trg truecase $modeldir/truecase-model.$train.$TRG"
    $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $workdir/$train.clean.unesc.tok.$TRG -model $modeldir/truecase-model.$train.$TRG
  fi
fi

# apply truecaser. skips if src\trg_tc=""
echo "start apply truecaser. skips if src\trg_tc="""
for prefix in "${apply_to[@]}"
 do
    if [ ! -f "$workdir/$prefix.unesc.tok$src_tc.$SRC" ]; then
      echo $workdir/$prefix.unesc.tok$src_tc.$SRC
      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.tok.$SRC -out $workdir/$prefix.unesc.tok.tc.$SRC -cmd "$mosesdecoder/scripts/recaser/truecase.perl < %s > %s -model $modeldir/truecase-model.$train.$SRC" -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.tc.$SRC
    fi
    if [ ! -f "$workdir/$prefix.unesc.tok$trg_tc.$TRG" ]; then
      echo $workdir/$prefix.unesc.tok$trg_tc.$TRG
      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.tok.$TRG -out $workdir/$prefix.unesc.tok.tc.$TRG -cmd "$mosesdecoder/scripts/recaser/truecase.perl < %s > %s -model $modeldir/truecase-model.$train.$TRG" -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.tc.$TRG
    fi
 done
echo "done apply truecaser. skips if src\trg_tc="""

echo "################### start BPE ###################"
# train BPE
if [ $train_models = true ]; then
  if [ ! -f "$modeldir/${SRC}_bpe.model" ]; then
    cat $workdir/$train.clean.unesc.tok$src_tc.$SRC | $subword_nmt/learn_bpe.py  -s $bpe_operations > $modeldir/${SRC}_bpe.model
  fi
  if [ ! -f "$modeldir/${TRG}_bpe.model" ]; then
    cat $workdir/$train.clean.unesc.tok$trg_tc.$TRG | $subword_nmt/learn_bpe.py  -s $bpe_operations > $modeldir/${TRG}_bpe.model
  fi
fi

# apply BPE
for prefix in "${apply_to[@]}"
 do
    if [ ! -f "$workdir/$prefix.unesc.tok$src_tc.bpe.$SRC" ]; then
      python $subword_nmt/apply_bpe.py --glossaries "=" -c $modeldir/${SRC}_bpe.model < $workdir/$prefix.unesc.tok$src_tc.$SRC > $workdir/$prefix.unesc.tok$src_tc.bpe.$SRC
    fi
    if [ ! -f "$workdir/$prefix.unesc.tok$trg_tc.bpe.$TRG" ]; then
      python $subword_nmt/apply_bpe.py --glossaries "=" -c $modeldir/${TRG}_bpe.model < $workdir/$prefix.unesc.tok$trg_tc.$TRG > $workdir/$prefix.unesc.tok$trg_tc.bpe.$TRG
    fi
 done
echo "done apply BPE"

if [ ! -f "$workdir/$train.clean.unesc.tok.bpe.$SRC$TRG" ]; then
   rm ${workdir}/tmp* -rf #
   rm ${workdir}/*.unesc.tok${src_tc}.${SRC} #
   rm $workdir/*.clean.unesc.tok.$SRC
   rm $workdir/*.cln.unesc.tok.$SRC
   rm $workdir/*.cln.unesc.$SRC
   rm $workdir/*.unesc.tok.$SRC #
   rm $workdir/*.unesc.$SRC #
   rm $workdir/*.cln.$SRC
   rm $workdir/*.clean.unesc.tok.$TRG
   rm $workdir/*.clean.unesc.tok${trg_tc}.$TRG
   rm $workdir/*.cln.unesc.tok.$TRG
   rm $workdir/*.cln.unesc.$TRG
   rm $workdir/*.unesc.tok.$TRG #
   rm $workdir/*.unesc.$TRG #
   rm $workdir/*.cln.$TRG
  echo "Removed non goal files"
fi
#cp $0 $workdir/preprocess.sh
echo "done"