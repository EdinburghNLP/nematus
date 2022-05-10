#!/bin/bash
set -e
echo "**************************************** in print_embedding_table.sh ****************************************"
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh ${language} ${debias_method}
model_type=bpe256
model_name=model.npz
model_dir=${debias_files_dir}/${language_dir}/scripts/models/${model_type}/${model_name}
#echo "model_dir: ${model_dir}"
output_filename=output_translate_${dst_language}.out.tmp
outputh_path=${debias_files_dir}/${language_dir}/output/not_important.txt
#echo "outputh_path: ${outputh_path}"
output_translate_path=${debias_files_dir}/${language_dir}/debias/output_translate_${dst_language}.txt
#echo "output_translate_path: ${output_translate_path}"
config="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 1, 'DEBIAS_METHOD': ${debias_method}}"

#echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m  ${model_dir} -k 12 -n -o ${outputh_path} -c ${config}"
exec > ${output_translate_path}
exec 2>&1
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m "$model_dir" \
     -k 12 -n -o "${outputh_path}" -c "${config}"
