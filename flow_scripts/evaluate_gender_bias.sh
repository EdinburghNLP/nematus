#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/evaluate_gender_bias-%j.out
echo "###############in evaluate_gender_bias.sh###############"
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh
#############preprocess###############
#echo "###############preprocess###############"
#echo "sh ${nematus_dir}/global_preprocess.sh ${dst_language}"
#sh ${nematus_dir}/global_preprocess.sh ${dst_language}

##############translate anti sentences to test gender bias###############
input_path=${nematus_dir}/${language_dir}/preprocess/anti.unesc.tok.tc.bpe.en
echo "input_path: ${input_path}"
model_type=bpe256
model_name=model.npz
model_dir=${nematus_dir}/${language_dir}/scripts/models/${model_type}/${model_name}
echo "model_dir: ${model_dir}"
#output_filename_debiased=debiased_anti_TEST.out.tmp
outputh_path_debiased=${nematus_dir}/${language_dir}/output/debiased_anti_${debias_method}.out.tmp
outputh_path_non_debiased=${nematus_dir}/${language_dir}/output/non_debiased_anti_${debias_method}.out.tmp
echo "outputh_path_debiased: ${outputh_path_debiased}"
echo "outputh_path_non_debiased: ${outputh_path_non_debiased}"
config_debiased="{'USE_DEBIASED': 1, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1, 'DEBIAS_METHOD': ${debias_method}}"
echo "config_debiased: ${config_debiased}"
config_non_debiased="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1, 'DEBIAS_METHOD': ${debias_method}}"
echo "###############translate anti debias###############"
echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m ${model_dir} -k 12 -n -o ${outputh_path_debiased} -c ${config_debiased}"
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m "$model_dir" \
     -k 12 -n -o "${outputh_path_debiased}" -c "${config_debiased}"
echo "###############translate anti non debias###############"
echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m ${model_dir} -k 12 -n -o ${outputh_path_non_debiased} -c ${config_non_debiased}"
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m "$model_dir" \
     -k 12 -n -o "${outputh_path_non_debiased}" -c "${config_non_debiased}"
#
###############merge translations###############
echo "###############merge translations###############"
python ${nematus_dir}/merge_translations.py \
     -c "{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0, 'DEBIAS_METHOD': ${debias_method}}" \
     -e 0
################prepare gender data###############
echo "###############prepare gender data###############"
python ${nematus_dir}/prepare_gender_data.py  -c "${config_non_debiased}"
############ gender evaluation###############
echo "###############gender evaluation###############"
output_result_path=${nematus_dir}/${language_dir}/debias/gender_evaluation_${dst_language}_${debias_method}.txt
exec > ${output_result_path}
exec 2>&1
cd /cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender
source venv/bin/activate
cd src
sh ../scripts/evaluate_debiased.sh $1 ${debias_method}




