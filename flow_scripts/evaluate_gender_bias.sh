echo "###############in evaluate_gender_bias.sh###############"
set -e
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh
###############preprocess###############
#echo "###############preprocess###############"
#sh ${nematus_dir}/${language_dir}/preprocess.sh

###############translate anti sentences to test gender bias###############
input_path=${nematus_dir}/${language_dir}/preprocess/anti.unesc.tok.tc.bpe.en
echo "input_path: ${input_path}"
model_type=bpe256
model_name=model.npz
model_dir=${nematus_dir}/${language_dir}/scripts/models/${model_type}/${model_name}
echo "model_dir: ${model_dir}"
output_filename_debiased=debiased_anti.out.tmp
#output_filename_debiased=debiased_anti_TEST.out.tmp
outputh_path_debiased=${nematus_dir}/${language_dir}/output/${output_filename_debiased}
output_filename_non_debiased=non_debiased_anti.out.tmp
outputh_path_non_debiased=${nematus_dir}/${language_dir}/output/${output_filename_non_debiased}
echo "outputh_path_debiased: ${outputh_path_debiased}"
echo "outputh_path_non_debiased: ${outputh_path_non_debiased}"
config_debiased="{'USE_DEBIASED': 1, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1}"
echo "config_debiased: ${config_debiased}"
config_non_debiased="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 1}"
#echo "###############translate anti debias###############"
#echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m ${model_dir} -k 12 -n -o ${outputh_path_debiased} -c ${config_debiased}"
#python ${nematus_dir}/nematus/translate.py \
#     -i "$input_path" \
#     -m "$model_dir" \
#     -k 12 -n -o "${outputh_path_debiased}" -c "${config_debiased}"
#echo "###############translate anti non debias###############"
#echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m ${model_dir} -k 12 -n -o ${outputh_path_non_debiased} -c ${config_non_debiased}"
#python ${nematus_dir}/nematus/translate.py \
#     -i "$input_path" \
#     -m "$model_dir" \
#     -k 12 -n -o "${outputh_path_non_debiased}" -c "${config_non_debiased}"

###############preparing translations to evaluation###############
echo "###############preparing translations to evaluation###############"
lines_to_keep=$(python ${nematus_dir}/prepare_eval.py \
     -c "{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'PRINT_LINE_NUMS': 0}" \
     -e 0)
################prepare gender data###############
echo "###############prepare gender data###############"
python ${nematus_dir}/prepare_gender_data.py  -c "${config_non_debiased}"
############## gender evaluation###############
echo "###############gender evaluation###############"
cd /cs/usr/bareluz/gabi_labs/nematus_clean/mt_gender
source venv/bin/activate
cd src
sh ../scripts/evaluate_debiased.sh $1 $dst_language




