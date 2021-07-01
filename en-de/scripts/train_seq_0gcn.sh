#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:4,vmem:8g
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/en-de_0gcn%j.out
#SBATCH --wckey=strmt

# module load cuda/10.0
# module load cudnn
# source /cs/snapless/oabend/borgr/envs/tf15/bin/activate

# module load tensorflow
module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate
# export CUDA_VISIBLE_DEVICES='0,1,2,3'

vocab_in=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.en 
vocab_out=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/vocab.clean.unesc.tok.tc.bpe.de
script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-de/scripts
echo "script_dir is ${script_dir}"
main_dir=$script_dir/../..
# data_dir=$script_dir/data
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/
model_dir=$script_dir/models
mkdir -p $model_dir

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

working_dir=$model_dir/0gcn
mkdir -p $working_dir

# json_bpe=$script_dir/data/conll14st-preprocessed.bpe.${src}${trg}.json
src_train=$data_dir/train.clean.unesc.tok.tc.bpe.de
trg_train=$data_dir/train.clean.unesc.tok.tc.bpe.en
src_bpe=$src_train.json
trg_bpe=$trg_train.json

# create dictionary if needed
if [ ! -f ${trg_bpe} ]; then
    echo "creating target dict"
    tmp="$working_dir/tmp_all_train"
    cat $vocab_out $trg_train $trg_dev> $tmp
    python $nematus_home/data/build_dictionary.py $tmp
    mv "$tmp.json" $trg_bpe
    rm $tmp
fi

if [ ! -f ${src_bpe} ]; then
    echo "creating source dict"
    tmp="$working_dir/tmp_all_train"
    cat $vocab_in $src_train $src_dev > $tmp
    python $nematus_home/data/build_dictionary.py $tmp
    mv "$tmp.json" $src_bpe
    rm $tmp
fi


src_dev=$data_dir/newstest2013.unesc.tok.tc.bpe.de
trg_dev=$data_dir/newstest2013.unesc.tok.tc.bpe.en

# train="conll14st-preprocessed"
# corpora=("en_esl-ud-dev.conllu" "en_esl-ud-test.conllu" "en_esl-ud-train.conllu")
len=40
batch_size=128
embedding_size=256
# token_batch_size=2048
# sent_per_device=4
tokens_per_device=162
dec_blocks=4
enc_blocks="${dec_blocks}"
lshw -C display | tail # write the acquired gpu properties

python3 $nematus_home/nematus/train.py \
    --source_dataset $src_train \
    --target_dataset $trg_train \
    --dictionaries $src_bpe $trg_bpe\
    --save_freq 30000 \
    --model $working_dir/model_seq_trans.npz \
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
    --valid_batch_size 4 \
    --max_tokens_per_device $tokens_per_device \
    --valid_freq 10000 \
    --disp_freq 1000 \
    --valid_script $script_dir/validate_seq.sh \
    --target_labels_num 0\
    --target_graph \
    --non_sequential \
    --target_gcn_layers 0 \
    --sample_freq 0 \
    --beam_freq 1000 \
    --beam_size 8 \
    --valid_remove_parse #&> /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/out$(date "+%Y.%m.%d-%H.%M.%S") &
    # --token_batch_size $token_batch_size \
    # --valid_token_batch_size $token_batch_size \
    # --print_per_token_pro /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/probs.last\
    # --max_sentences_per_device $sent_per_device \
    # --tie_encoder_decoder_embeddings \
    # --tie_decoder_embeddings \
    # --token_batch_size 16384 \
    # --batch_size 256 \

echo done