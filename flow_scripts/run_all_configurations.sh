#!/bin/bash
set -e


SHORT=c,p,t,h
LONG=collect_embedding_table,preprocess,translate,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

collect_embedding_table=""
preprocess=""
translate=""

while :
do
  case "$1" in
    -c | --collect_embedding_table )
      collect_embedding_table="-c"
      shift 1
      ;;
    -p | --preprocess )
      preprocess="-p"
      shift 1
      ;;
    -t | --translate )
      translate="-t"
      shift 1
      ;;
    -h | --help)
      echo "usage:
Optional arguments:
  -c, --collect_embedding_table   collect embedding table .
  -p, --preprocess                preprocess the anti dataset .
  -t, --translate                 translate the entire dataset .
  -h, --help                      help message ."
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      exit 1;;
  esac
done
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh ${language} ${debias_method}

echo "#################### cleanup ####################"
nematus_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus
python ${debias_files_dir}/cleanup.py ${collect_embedding_table} ${translate}

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l de -d 0 ${collect_embedding_table} ${preprocess} ${translate}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l de -d 1 ${collect_embedding_table} ${preprocess} ${translate}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l he -d 0 ${collect_embedding_table} ${preprocess} ${translate}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l he -d 1 ${collect_embedding_table} ${preprocess} ${translate}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l ru -d 0 ${collect_embedding_table} ${preprocess} ${translate}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l ru -d 1 ${collect_embedding_table} ${preprocess} ${translate}

echo "#################### write results to table ####################"
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus_env3/bin/activate
python ${debias_files_dir}/write_results_to_table.py