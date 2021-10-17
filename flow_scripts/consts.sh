set -e
#module load tensorflow/2.0.0
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus/nematus_env3/bin/activate
export src_language=en
export dst_language=ru
echo "language: ${dst_language}"
# set PYTHONPATH
export cur_dir=`pwd`
echo "cur dir: ${cur_dir}"
export PYTHONPATH=${PYTHONPATH}:${cur_dir}
echo "PYTHONPATH: ${PYTHONPATH}"
# set up parameters
export nematus_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus
export language_dir=${src_language}-${dst_language}