#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c1
#SBATCH --time=2-0
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=omry.gilon@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/en-he_translate%j.out



module load tensorflow/2.0.0
source /cs/snapless/oabend/borgr/envs/tg/bin/activate

script_dir=`dirname $0`
script_dir=/cs/snapless/oabend/borgr/TG/en-he/scripts/
echo "script_dir is ${script_dir}"
main_dir=$script_dir/../..

#language-independent variables (toolkit locations)
. $script_dir/../vars
src=he
trg=en
model_type="bpe256"
output_path=/cs/snapless/oabend/borgr/TG/en-he/output/tmp_omrygilon2.out
input_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/20.07.21/omrygilon_generalization/test_he.tok.he
model_name="model.npz"

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

model_dir=$script_dir/models/bpe256/
tmp_path=$output_path.tmp

python "$nematus_home"/nematus/translate.py \
     -i "$input_path" \
     -m  "$model_dir/$model_name" \
     -k 12 -n -o "$tmp_path"
$script_dir/postprocess.sh < "$tmp_path" > "$output_path"

preprocessed_path="${tmp_path}.proc"
$script_dir/postprocess.sh < "${tmp_path}" > "${preprocessed_path}"

echo python $remove_edges "${preprocessed_path}" -o "${output_path}"
python "$nematus_home"/nematus/parsing/remove_edges.py "${preprocessed_path}" -o "${output_path}"
rm $preprocessed_path $tmp_path
echo translated to "${output_path}"
