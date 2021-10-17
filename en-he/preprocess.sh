#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c4
#SBATCH --time=2-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/preprocess-en_he%j.out

source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/nematus_env3/bin/activate

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # fails on symlinks
SCRIPT_FILE=`basename "$0"`

# suffix of language files
SRC=en

TRG=he

cpus=4

#!#!#!# extras config #!#!#!#
# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=30000
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/cs/snapless/oabend/borgr/SSMT/preprocess/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/cs/snapless/oabend/borgr/SSMT/preprocess/subword-nmt

# path to nematus repository: https://github.com/borgr/nematus
nematus_home=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus

# path to unescaping scripts
unescape=/cs/snapless/oabend/borgr/SSMT/preprocess/unescape.py

# dirs
datadir=/cs/snapless/oabend/borgr/SSMT/data/${SRC}_${TRG}/

workdir="/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/${SRC}_${TRG}/20.07.21"
mkdir -p "$workdir"

# copy script file to dir
cp "${SCRIPT_DIR}/${SCRIPT_FILE}" "${workdir}/${SCRIPT_FILE}"

modeldir=/cs/snapless/oabend/borgr/SSMT/preprocess/model/${SRC}_${TRG}/20.07.21
mkdir -p "$modeldir"

# corpora
train=train
apply_to=(dev test)

# check lengths match in train
#srclen=$(cat $datadir/$train.$SRC | wc -l)
#trglen=$(cat $datadir/$train.$TRG | wc -l)
#if [ $srclen != $trglen ] ; then
#  echo "$datadir/$train.$LANG does not match the length of $datadir/$train.$TRG"
#  exit 1
#fi
#
# check lengths match in non-train corpora
#for prefix in "${apply_to[@]}"
#  do
#  srclen=$(cat $datadir/$prefix.$SRC | wc -l)
#  trglen=$(cat $datadir/$prefix.$TRG | wc -l)
#  if [ $srclen != $trglen ] ; then
#    echo "$datadir/$prefix.$LANG does not match the length of $datadir/$prefix.$TRG"
#    exit 1
#  fi
#done

# remove weird characters and insert ##AT##-##AT##
if [ ! -f "$workdir/${train}.cln.$SRC" ]; then
    python /cs/snapless/oabend/borgr/SSMT/preprocess/preprocess.py -s $SRC -f $datadir -n "${train}" -o "${train}.cln" -d $workdir
fi
#if [ ! -f "$workdir/$train.cln.$TRG" ]; then
#    # remove lines dropped in target
#    python /cs/snapless/oabend/borgr/SSMT/preprocess/subsample_lines.py  -i $datadir/$train.$TRG -o $workdir/$train.cln.$TRG -d "${workdir}/${train}.idx"
#fi
#if [ ! -f "$workdir/$train.cln.$TRG" ]; then
#    # remove lines dropped in target
#    echo "failed to create $workdir/$train.cln.$TRG"
#    exit 1
#fi

# leave original copy at the base data dir
for prefix in "${apply_to[@]}" 
 do
    if [ ! -f "$workdir/$prefix.$SRC" ]; then
        cp "$datadir/$prefix.$SRC" "$workdir/$prefix.$SRC"
    fi
#    if [ ! -f "$workdir/$prefix.$TRG" ]; then
#        cp "$datadir/$prefix.$TRG" "$workdir/$prefix.$TRG"
#    fi
 done

apply_to+=("${train}.cln")

# tokenize
for prefix in ${apply_to[@]}
 do
    if [ ! -f $workdir/$prefix.unesc.tok.$SRC ] ; then
      echo $SRC $prefix 
      python $unescape $workdir/$prefix.$SRC
      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.$SRC -out $workdir/$prefix.unesc.tok.$SRC -cmd "$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl $SRC < %s |  
      $mosesdecoder/scripts/tokenizer/tokenizer_PTB.perl -l $SRC |
      $mosesdecoder/scripts/tokenizer/deescape-special-chars-PTB.perl > %s " -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.$SRC
    fi
  if [ ! -f $workdir/$prefix.unesc.tok.$SRC ] ; then
      exit 1
  fi

#    if [ ! -f $workdir/$prefix.unesc.tok.$TRG ] ; then
#      python $unescape $workdir/$prefix.$TRG
#      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.$TRG -out $workdir/$prefix.unesc.tok.$TRG -cmd "$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl $TRG < %s |
#      $mosesdecoder/scripts/tokenizer/tokenizer_PTB.perl -l $TRG |
#      $mosesdecoder/scripts/tokenizer/deescape-special-chars-PTB.perl > %s " -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.$TRG
#    fi
 done

#if [ ! -f "$workdir/$train.cln.unesc.tok.$TRG" ]; then
#    echo "$workdir/$train.cln.unesc.tok.$TRG not found, please run preprocess on \" $TRG \" to create it, before rerunning this script"
#    exit 1
#fi

#clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
if [ ! -f "$workdir/$train.clean.unesc.tok.$SRC" ]; then
  $mosesdecoder/scripts/training/clean-corpus-n.perl $workdir/$train.cln.unesc.tok $SRC $TRG $workdir/$train.clean.unesc.tok 1 50
fi

apply_to[-1]="${train}.clean"

# train truecaser
if [ ! -f "$modeldir/truecase-model.$train.$SRC" ]; then
  $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $workdir/$train.clean.unesc.tok.$SRC -model $modeldir/truecase-model.$train.$SRC
fi

# apply truecaser 
for prefix in "${apply_to[@]}"
 do
    if [ ! -f "$workdir/$prefix.unesc.tok.tc.$SRC" ]; then
      echo $workdir/$prefix.unesc.tok.tc.$SRC
      $mosesdecoder/scripts/ems/support/generic-multicore-parallelizer.perl -in $workdir/$prefix.unesc.tok.$SRC -out $workdir/$prefix.unesc.tok.tc.$SRC -cmd "$mosesdecoder/scripts/recaser/truecase.perl < %s > %s -model $modeldir/truecase-model.$train.$SRC" -cores $cpus -tmpdir $workdir/tmp.$prefix.unesc.tok.tc.$SRC
    fi
 done

# train BPE
if [ ! -f "$modeldir/${SRC}_bpe.model" ]; then
  cat $workdir/$train.clean.unesc.tok.tc.$SRC | $subword_nmt/learn_bpe.py  -s $bpe_operations > $modeldir/${SRC}_bpe.model
fi
#if [ ! -f "$modeldir/${TRG}_bpe.model" ]; then
#  cat $workdir/$train.clean.unesc.tok.$TRG | $subword_nmt/learn_bpe.py  -s $bpe_operations > $modeldir/${TRG}_bpe.model
#fi

# apply BPE
for prefix in "${apply_to[@]}"
 do
    if [ ! -f "$workdir/$prefix.unesc.tok.tc.bpe.$SRC" ]; then
      python $subword_nmt/apply_bpe.py --glossaries "=" -c $modeldir/${SRC}_bpe.model < $workdir/$prefix.unesc.tok.tc.$SRC > $workdir/$prefix.unesc.tok.tc.bpe.$SRC
    fi
#    if [ ! -f "$workdir/$prefix.unesc.tok.bpe.$TRG" ]; then
#      python $subword_nmt/apply_bpe.py --glossaries "=" -c $modeldir/${TRG}_bpe.model < $workdir/$prefix.unesc.tok.$TRG > $workdir/$prefix.unesc.tok.bpe.$TRG
#    fi
 done

python $subword_nmt/get_vocab.py < $workdir/$train.clean.unesc.tok.tc.bpe.$SRC > $workdir/vocab.clean.unesc.tok.tc.bpe.$SRC
#python $subword_nmt/get_vocab.py < $workdir/$train.clean.unesc.tok.bpe.$TRG > $workdir/vocab.clean.unesc.tok.bpe.$TRG
# # # python $subword_nmt/get_vocab.py < $workdir/$train.clean.unesc.tok.tc.bpe.$TRG > $workdir/vocab.clean.unesc.tok.tc.bpe.$TRG

# # # cat $workdir/$train.clean.unesc.tok.tc.bpe.$SRC | $subword_nmt/get_vocab.py  > $workdir/vocab.clean.unesc.tok.tc.bpe.$SRC

# # # cat $workdir/$train.clean.unesc.unesc.tok.tc.bpe.$SRC >  $workdir/$train.clean.unesc.unesc.tok.tc.bpe.$SRC

## For nematus compatibility
#cat "$workdir/$train.clean.unesc.tok.tc.bpe.$SRC" "$workdir/$train.clean.unesc.tok.bpe.$TRG" > "$workdir/$train.clean.unesc.tok.bpe.$SRC$TRG"

#python /cs/snapless/oabend/borgr/SSMT/vocab.py -c $workdir/config_vocab.yaml

# switch to nematus environment (py3)
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/nematus_env3/bin/activate

# build network dictionary
python $nematus_home/data/build_dictionary.py "$workdir/$train.clean.unesc.tok.tc.bpe.$SRC"


if [ -f "$workdir/$train.clean.unesc.tok.bpe.$SRC$TRG" ]; then
   rm ${workdir}/tmp* -rf
#   rm $workdir/*.clean.unesc.tok.$SRC
#   rm $workdir/*.unesc.tok.$SRC
#   rm $workdir/*.unesc.$SRC
#   rm $workdir/*.$SRC
#   rm $workdir/*.clean.unesc.tok.$TRG
#   rm $workdir/*.unesc.tok.$TRG
#   rm $workdir/*.unesc.$TRG
#   rm $workdir/*.$TRG
  echo "Removed non goal files"
fi

cp $0 $workdir/preprocess.sh

echo "done"