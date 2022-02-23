#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/run_all_flows-%j.out
echo $SHELL

#sh print_embedding_table.sh $1 $2
sh evaluate_gender_bias.sh $1 $2
sh evaluate_translation.sh $1 $2
