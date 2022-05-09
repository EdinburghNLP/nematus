#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/en-he_translate%j.out



#module load tensorflow/2.0.0
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus_env3/bin/activate

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/scripts/
echo "script_dir is ${script_dir}"
main_dir=$script_dir/../..

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

#model_type="0gcn"
model_type="bpe256"
#output_path=/cs/snapless/oabend/borgr/TG/en-he/output/tmp.out
output_path=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/output/tmp.out
input_path=$data_dir/test.unesc.tok${src_tc}.bpe.${src}
#model_name=model_seq_trans.npz
model_name=model.npz

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

model_dir=$script_dir/models/${model_type}/
tmp_path=$output_path.tmp

echo "python ${nematus_home}/nematus/translate.py \n
     -i ${input_path} \n
     -m  ${model_dir}/${model_name} \n
     -k 12 -n -o ${tmp_path}"

python "$nematus_home"/nematus/translate.py \
     -i "$input_path" \
     -m  "$model_dir/$model_name" \
     -k 12 -n -o "$tmp_path" -d 0
$script_dir/postprocess.sh < "$tmp_path" > "$output_path"

preprocessed_path="${tmp_path}.proc"
$script_dir/postprocess.sh < "${tmp_path}" > "${preprocessed_path}"

echo python $remove_edges "${preprocessed_path}" -o "${output_path}"
python $remove_edges "${preprocessed_path}" -o "${output_path}"
rm $preprocessed_path $tmp_path
echo translated to "${output_path}"