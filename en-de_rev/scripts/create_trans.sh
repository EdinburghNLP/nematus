#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/create_trans%j.out

source /cs/snapless/oabend/borgr/envs/tg/bin/activate
script=/cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py
out_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/

conllu_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2013.de.conllu
bpe_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2013.unesc.tok.tc.bpe.de
#conllu_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/train.clean.unesc.tok.tc.conllu.de
#bpe_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.de

python $script --conllu $conllu_path --bpe $bpe_path --out $out_dir
echo done