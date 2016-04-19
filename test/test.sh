#!/bin/bash

nematus_path=../

export THEANO_FLAGS=device=cpu,floatX=float32

python $nematus_path/nematus/translate.py -p 1 \
	-m model.npz  \
	-i newstest2011.en \
	-o ./newstest2011.trans.de \
        -n \
        -k 12



