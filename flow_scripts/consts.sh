set -e
#module load tensorflow/2.0.0
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus_env3/bin/activate
export src_language=en
export dst_language=$1
export debias_method=$2
#echo "language: ${dst_language}"
#echo "debias_method: ${debias_method}"
# set PYTHONPATH
export project_dir=/cs/usr/bareluz/gabi_labs/nematus_clean
#echo "project_dir: ${project_dir}"
export PYTHONPATH=${PYTHONPATH}:${project_dir}
#echo "PYTHONPATH: ${PYTHONPATH}"sh parameters
# set up parameters
export nematus_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus
export debias_files_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/debias_files
export snapless_data_dir=/cs/snapless/gabis/bareluz
export language_dir=${src_language}-${dst_language}

leshem_data_path=/cs/snapless/oabend/borgr/SSMT/preprocess/data
case ${dst_language} in
	ru)
		export input_path=${leshem_data_path}/${src_language}_${dst_language}/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en
		export language_num=0
		;;
	de)
		export input_path=${leshem_data_path}/${src_language}_${dst_language}/5.8/newstest2012.unesc.tok.tc.bpe.en
		export language_num=1
		;;
	he)
		export input_path=${leshem_data_path}/${src_language}_${dst_language}/20.07.21/dev.unesc.tok.tc.bpe.en
		export language_num=2
		;;
	*)
		echo "invalid language given. the possible languages are ru, de, he"
		;;
esac
