#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c1
#SBATCH --time=2-0
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/en-de_rev_translate%j.out


module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate

script_dir=`dirname $0`
script_dir=/cs/snapless/oabend/borgr/TG/en-de_rev/scripts/
echo "script_dir is ${script_dir}"
main_dir=$script_dir/../..
# data_dir=$script_dir/data
data_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

model_name="0gcn"
output_path=/cs/snapless/oabend/borgr/TG/en-de/output/tmp.out
input_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2014.unesc.tok.tc.bpe.${src}

if [ ! -z "$1" ]
  then
  model_type=$1
fi
if [ ! -z "$2" ]
  then
  output_path=$2
fi
if [ ! -z "$3" ]
  then
  input_path=$3
fi

if [ ! -z "$4" ]
    then
    model_name=$4
fi
echo model_name $model_name
echo output_path $output_path
echo input_path $input_path
echo model_type $model_type
model_dir=$script_dir/models/${model_type}/
tmp_path=$output_path.tmp

echo "python $nematus_home/nematus/translate.py -i $input_path -m  $model_dir/$model_name -k 12 -n -o $tmp_path $script_dir/postprocess.sh < $tmp_path > $output_path"

python "${nematus_home}"/nematus/translate.py \
     -i "${input_path}" \
     -m  "${model_dir}/${model_name}" \
     -k 12 -n -o "${tmp_path}"

preprocessed_path="${tmp_path}.proc"
$script_dir/postprocess.sh < "${tmp_path}" > "${preprocessed_path}"

echo python $remove_edges "${preprocessed_path}" -o "${output_path}"
python $remove_edges "${preprocessed_path}" -o "${output_path}"
rm $preprocessed_path $tmp_path
echo translated to "${output_path}"