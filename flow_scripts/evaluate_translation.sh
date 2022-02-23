#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/evaluate_translation-%j.out
echo "###############in evaluate_translation.sh###############"
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh

###############translate some dataset to test translation quality ###############
echo "input_path: ${input_path}"
model_type=bpe256
model_name=model.npz
model_dir=${nematus_dir}/${language_dir}/scripts/models/${model_type}/${model_name}
echo "model_dir: ${model_dir}"
outputh_path_debiased=${nematus_dir}/${language_dir}/output/debiased_${debias_method}.out.tmp
outputh_path_non_debiased=${nematus_dir}/${language_dir}/output/non_debiased_${debias_method}.out.tmp
echo "outputh_path_debiased: ${outputh_path_debiased}"
echo "outputh_path_non_debiased: ${outputh_path_non_debiased}"
config_debiased="{'USE_DEBIASED': 1, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1, 'DEBIAS_METHOD': ${debias_method}}"
config_non_debiased="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1, 'DEBIAS_METHOD': ${debias_method}}"

###############translate debiased###############
echo "###############translate debiased###############"
echo "python ${nematus_dir}/nematus/translate.py -i $input_path -m  $model_dir -k 12 -n -o ${outputh_path_debiased} -c ${config_debiased}"
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m "$model_dir" \
     -k 12 -n -o "${outputh_path_debiased}" -c "${config_debiased}"

###############translate non debiased###############
echo "###############translate non debiased###############"
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m  "$model_dir" \
     -k 12 -n -o "${outputh_path_non_debiased}" -c "${config_non_debiased}"

###############merge_translations###############
echo "###############merge_translations###############"
python ${nematus_dir}/merge_translations.py \
     -c "{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0, 'DEBIAS_METHOD': ${debias_method}}" \
     -e 1
##############evaluate translation quality###############
echo "###############evaluate translation quality###############"
output_result_path=${nematus_dir}/${language_dir}/debias/translation_evaluation_${dst_language}_${debias_method}.txt
exec > ${output_result_path}
exec 2>&1
python ${project_dir}/mt_gender/src/evaluate_translation.py \
     -c "{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0, 'DEBIAS_METHOD': ${debias_method}}"