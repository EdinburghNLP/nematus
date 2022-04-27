#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/run_all_flows-%j.out

SHORT=l:,d:,c,p,t,h
LONG=language:,debias_method:,collect_embedding_table,preprocess,translate,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

collect_embedding_table=false
preprocess=""
translate=""

while :
do
  case "$1" in
    -l | --language )
      language="$2"
      shift 2
      ;;
    -d | --debias_method )
      debias_method="$2"
      shift 2
      ;;
    -c | --collect_embedding_table )
      collect_embedding_table=true
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
Mandatory arguments:
  -l, --language                  the destination translation language .
  -d, --debias_method             the debias method .
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

#echo "collect_embedding_table: ${collect_embedding_table},preprocess: ${preprocess},translate: ${translate}"

scripts_dir=`pwd`
source ${scripts_dir}/consts.sh ${language} ${debias_method}

if [ $collect_embedding_table = true ]; then
  sh print_embedding_table.sh ${language} ${debias_method}
fi
sh evaluate_gender_bias.sh -l ${language} -d ${debias_method} ${preprocess} ${translate}
sh evaluate_translation.sh -l ${language} -d ${debias_method} ${translate}
