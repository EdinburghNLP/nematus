#!/bin/bash
#SBATCH --mem=64g
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/validate_all%j.out
#SBATCH -c20
#SBATCH --wckey=strmt

module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate
python /cs/snapless/oabend/borgr/TG/nematus/validate_all.py #--force
#SBATCH -c4
#SBATCH --gres=gpu:4,vmem:10g
#SBATCH --gres=gpu:4

