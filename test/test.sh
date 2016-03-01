#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=cpu,floatX=float32

cd $PBS_O_WORKDIR
python ./translate.py -n -p 8 \
	$HOME/models/model_session2.npz  \
	$HOME/data/europarl-v7.fr-en.en.tok.pkl \
	$HOME/data/europarl-v7.fr-en.fr.tok.pkl \
	$HOME/data/newstest2011.en.tok \
	./newstest2011.trans.fr.tok



