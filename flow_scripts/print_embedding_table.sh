set -e
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh
dataset=newstest2019-enru.unesc.tok.tc.bpe.en
input_path=${nematus_dir}/${language_dir}/evaluate/newstest2019-${src_language}${dst_language}.unesc.tok.tc.bpe.${src_language}
echo "input_path: ${input_path}"
model_type=bpe256
model_name=model.npz
model_dir=${nematus_dir}/${language_dir}/scripts/models/${model_type}/${model_name}
echo "model_dir: ${model_dir}"
output_filename=test1.out.tmp
outputh_path=${nematus_dir}/${language_dir}/output/${output_filename}
echo "outputh_path: ${outputh_path}"
output_translate_path=${nematus_dir}/${language_dir}/debias/output_translate_${dst_language}.txt

echo "python ${nematus_dir}/nematus/translate.py -i ${input_path} -m  ${model_dir} -k 12 -n -o ${outputh_path} -d 0 -l 0 -c 1"
exec > ${output_translate_path}
exec 2>&1
python ${nematus_dir}/nematus/translate.py \
     -i "$input_path" \
     -m  "$model_dir" \
     -k 12 -n -o "${outputh_path}" -d 0 -l 0 -c 1

