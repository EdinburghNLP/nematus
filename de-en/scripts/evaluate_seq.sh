#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c1
#SBATCH --time=2-0
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus/slurm/evaluate%j.out
#SBATCH --wckey=strmt
module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate

main_dir=/cs/usr/bareluz/gabi_labs/nematus/de-en/
# data_dir=$main_dir/data
# data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/
#data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD

#language-independent variables (toolkit locations)
. $main_dir/vars

#language-dependent variables (source and target language)
. $main_dir/vars

remove_edges=$nematus_home/nematus/parsing/remove_edges.py
translations=$1
ref=$2

python $remove_edges $translations |\
    $nematus_home/data/multi-bleu-detok.perl $ref | \
    cut -f 3 -d ' ' | \
    cut -f 1 -d ','
