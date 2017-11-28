#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=gpu,floatX=float32

cd $PBS_O_WORKDIR
python ./train_lm.py



