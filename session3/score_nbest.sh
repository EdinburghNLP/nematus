#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=cpu,floatX=float32

cd $PBS_O_WORKDIR
python ./rescore_with_lm.py -n -b 0.5 \
	${HOME}/models/model_session0.npz \
	${HOME}/models/model_session0.npz.pkl \
	${HOME}/data/wiki.tok.txt.gz.pkl \
	${HOME}/data/europarl-v7.fr-en.en.tok.pkl \
	./newstest2011.trans.en.tok \
    ./newstest2011.trans.en.tok.rescored

