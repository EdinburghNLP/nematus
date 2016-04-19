#!/bin/sh

S=fr
T=en

dev=newstest2011.$S.tok.bpe
ref=newstest2011.$T.tok
prefix=model.npz
nematus_path=../

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu,on_unused_input=warn time python $nematus_path/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev -o $dev.output.dev -k 12 -n -p 2


$nematus_path/data/postprocess.sh < $dev.output.dev > $dev.output.postprocessed.dev


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$nematus_path/data/multi-bleu.perl $ref < $dev.output.postprocessed.dev >> ${prefix}_bleu_scores
BLEU=`$nematus_path/multi-bleu.perl $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi

