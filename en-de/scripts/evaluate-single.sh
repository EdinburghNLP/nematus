#!/bin/bash
#SBATCH --mem=16g
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=log/slurm/evaluate_nem-%j.out

#SBATCH -c8
# module load tensorflow
# module load theano
# source /cs/snapless/oabend/borgr/envs/nematus/bin/activate
# source /cs/snapless/oabend/borgr/envs/pytorch/bin/activate 

reqsubstr="bla"
if [ -z "${hostname##*$reqsubstr*}" ]; then
    source /cs/snapless/oabend/borgr/envs/blaise2/bin/activate 
fi

model_dir=/cs/snapless/oabend/borgr/locality/nematus/scripts/wmt17_systems/en-de/
cd $model_dir
echo $model_dir

main_dir=$script_dir/..
working_dir=$model_dir/output/
data_dir=/cs/snapless/oabend/borgr/locality/data/challenge/
nematus_data=/cs/snapless/oabend/borgr/locality/data/nematus/
if [ ! -d $working_dir ]; then
    mkdir $working_dir
fi

#language-independent variables (toolkit locations)
. $model_dir/../vars

#language-dependent variables (source and target language)
. $model_dir/vars

if [ -z $1 ]; then
    test_prefix=newstest2013
else
    test_prefix=$1
fi

# don't use all data
if [ ! -z $2 ]; then
    head $data_dir/$test_prefix.$src -n $2 > $nematus_data/${test_prefix}_$2.$src
    head $nematus_data/$test_prefix.bpe.$src -n $2 > $nematus_data/${test_prefix}_$2.bpe.$src
    if [ -s $nematus_data/$test_prefix.tok.$trg ]; then
        echo using$nematus_data/$test_prefix.tok.$trg
        head $nematus_data/$test_prefix.tok.$trg -n $2 > $nematus_data/${test_prefix}_$2.$trg
    else
        echo using$data_dir/$test_prefix.$trg
        echo not using$data_dir/$test_prefix.tok.$trg
        head $data_dir/$test_prefix.$trg -n $2 > $nematus_data/${test_prefix}_$2.$trg
    fi
    test_prefix=${test_prefix}_$2
fi

test=$test_prefix.bpe.$src
ref=$test_prefix.$trg

echo "test_prefix $test_prefix"

echo "should write to $working_dir/$test.output"
echo $src "->" $trg 
echo input $nematus_data/$test
echo ref  $nematus_data/$ref
# translate if needed
if [ ! -s $working_dir/$test.output ]; then
    # $model_dir/preprocess.sh $nematus_data/$test_prefix.$src| \
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device python $nematus_home/nematus/translate.py \
         -m $model_dir/model.l2r.ens1.npz \
         -i $nematus_data/$test \
         -k 12 -n > $working_dir/$test.output ;

    echo "Finished translating postprocessing..."
else
    echo "reading translations from $working_dir/$test.output"
fi

$model_dir/postprocess.sh < $working_dir/$test.output > $working_dir/$test.output.postprocessed

echo "Calculating BLEU..."
# evaluate with detokenized BLEU (same as mteval-v13a.pl)
$nematus_home/data/multi-bleu-detok.perl $nematus_data/$ref < $working_dir/$test.output.postprocessed
