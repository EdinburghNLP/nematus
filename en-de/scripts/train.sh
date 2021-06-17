#!/bin/bash
#SBATCH --mem=64g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:4
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus/slurm/en-de%j.out

# module load cuda/10.0
# module load cudnn
# # module unload tensorflow
# source /cs/snapless/oabend/borgr/envs/tf15/bin/activate
# # module load tensorflow/2.0.0
# # source /cs/snapless/oabend/borgr/envs/p37/bin/activate
# export CUDA_VISIBLE_DEVICES='0,1,2,3'

#module load tensorflow/2.0.0
#source /cs/snapless/oabend/borgr/envs/tg/bin/activate
source /cs/usr/bareluz/gabi_labs/nematus/nematus_env3/bin/activate

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus/en-de/scripts/
echo "script_dir is ${script_dir}"
main_dir=$script_dir/../..
# data_dir=$script_dir/data
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/
model_dir=$script_dir/models/
mkdir -p $model_dir

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

working_dir=$model_dir/bpe256

src_train=$data_dir/train.clean.unesc.tok.tc.bpe.de
trg_train=$data_dir/train.clean.unesc.tok.tc.bpe.en
src_bpe=$src_train.json
trg_bpe=$trg_train.json

if [ ! -f ${trg_bpe} ]; then
    python $nematus_home/data/build_dictionary.py $trg_train
fi

if [ ! -f ${src_bpe} ]; then
    python $nematus_home/data/build_dictionary.py $src_train
fi


src_dev=$data_dir/newstest2013.unesc.tok.tc.bpe.en
trg_dev=$data_dir/newstest2013.unesc.tok.tc.bpe.de

len=40
batch_size=128
embedding_size=256
# token_batch_size=2048
# sent_per_device=4
tokens_per_device=162
dec_blocks=4
enc_blocks="${dec_blocks}"
lshw -C display | tail # write the acquired gpu properties

echo "nematus_home is: ${nematus_home}"


echo "now bar the horny starts ;)"
echo "python3 ${nematus_home}/nematus/train.py \n
    --source_dataset ${src_train} \n
    --target_dataset ${trg_train} \n
    --dictionaries ${src_bpe} ${trg_bpe} \n
    --save_freq 10000
    --model ${working_dir}/model.npz \n
    --reload latest_checkpoint \n
    --model_type transformer \n
    --embedding_size ${embedding_size} \n
    --state_size ${embedding_size} \n
    --loss_function per-token-cross-entropy \n
    --label_smoothing 0.1 \n
    --optimizer adam \n
    --adam_beta1 0.9 \n
    --adam_beta2 0.98 \n
    --adam_epsilon 1e-09 \n
    --transformer_dec_depth ${dec_blocks} \n
    --transformer_enc_depth ${enc_blocks} \n
    --learning_schedule transformer \n
    --warmup_steps 4000 \n
    --maxlen ${len} \n
    --batch_size ${batch_size} \n
    --translation_maxlen ${len} \n
    --normalization_alpha 0.6 \n
    --valid_source_dataset ${src_dev} \n
    --valid_target_dataset ${trg_dev} \n
    --valid_batch_size 120 \n
    --valid_token_batch_size 4096 \n
    --valid_freq 10000 \n
    --valid_script ${script_dir}/validate.sh \n

    --disp_freq 1000 \n
    --sample_freq 0 \n
    --beam_freq 1000 \n
    --beam_size 4 \n
    --translation_maxlen ${len}
"

echo "now bar the horny came "

##############run with full dataset##############
python3 $nematus_home/nematus/train.py \
    --source_dataset $src_train \
    --target_dataset $trg_train \
    --dictionaries $src_bpe $trg_bpe\
    --save_freq 5000 \
    --model $working_dir/model.npz \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size $embedding_size \
    --state_size $embedding_size \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --transformer_dec_depth $dec_blocks \
    --transformer_enc_depth $enc_blocks \
    --learning_schedule transformer \
    --warmup_steps 4000 \
    --maxlen $len \
    --batch_size $batch_size \
    --translation_maxlen $len \
    --normalization_alpha 0.6 \
    --valid_source_dataset $src_dev \
    --valid_target_dataset $trg_dev \
    --valid_batch_size 120 \
    --valid_token_batch_size 4096 \
    --valid_freq 10000 \
    --valid_script $script_dir/validate.sh \
    --disp_freq 1000 \
    --sample_freq 0 \
    --beam_freq 1000 \
    --beam_size 4 \
    --translation_maxlen $len

    # --token_batch_size 8192 \
    # --tie_encoder_decoder_embeddings \
    # --tie_decoder_embeddings \
    # --token_batch_size 16384 \
    # --batch_

##############run with small dataset##############
#    $nematus_home/nematus/train.py \
#    --source_dataset /cs/usr/bareluz/gabi_labs/nematus/nematus/en_small_data \
#    --target_dataset /cs/usr/bareluz/gabi_labs/nematus/nematus/de_small_data \
#    --dictionaries $src_bpe $trg_bpe\
#    --save_freq 10000 \
#    --model $working_dir/model.npz \
#    --reload latest_checkpoint \
#    --model_type transformer \
#    --embedding_size $embedding_size \
#    --state_size $embedding_size \
#    --loss_function per-token-cross-entropy \
#    --label_smoothing 0.1 \
#    --optimizer adam \
#    --adam_beta1 0.9 \
#    --adam_beta2 0.98 \
#    --adam_epsilon 1e-09 \
#    --transformer_dec_depth $dec_blocks \
#    --transformer_enc_depth $enc_blocks \
#    --learning_schedule transformer \
#    --warmup_steps 4000 \
#    --maxlen $len \
#    --batch_size $batch_size \
#    --translation_maxlen $len \
#    --normalization_alpha 0.6 \
#    --valid_source_dataset $src_dev \
#    --valid_target_dataset $trg_dev \
#    --valid_batch_size 120 \
#    --valid_token_batch_size 4096 \
#    --valid_freq 10000 \
#    --valid_script $script_dir/validate.sh \
#    --disp_freq 1000 \
#    --sample_freq 0 \
#    --beam_freq 1000 \
#    --beam_size 4 \
#    --translation_maxlen $len

    # --token_batch_size 8192 \
    # --tie_encoder_decoder_embeddings \
    # --tie_decoder_embeddings \
    # --token_batch_size 16384 \
    # --batch_size 256 \


echo done