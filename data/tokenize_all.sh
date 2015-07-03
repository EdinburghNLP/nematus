#!/bin/bash

for F in `ls ./training/* | grep -v pkl | grep -v tok`
do
    echo "perl ./tokenizer.perl -l ${F:(-2)} < $F > $F.tok"
    perl ./tokenizer.perl -l ${F:(-2)} < $F > $F.tok
done

for F in `ls ./dev/*.?? | grep -v tok`
do
    echo "perl ./tokenizer.perl -l ${F:(-2)} < $F > $F.tok"
    perl ./tokenizer.perl -l ${F:(-2)} < $F > $F.tok
done
