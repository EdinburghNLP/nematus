#!/bin/bash
set -e
SHORT=l:,d:,c,p,t,h
LONG=language:,debias_method:,collect_embedding_table,preprocess,translate,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

collect_embedding_table=""
preprocess=false
translate=false

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
      collect_embedding_table="-c"
      shift 1
      ;;
    -p | --preprocess )
      preprocess=true
      shift 1
      ;;
    -t | --translate )
      translate=true
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

echo "collect_embedding_table $collect_embedding_table"
if [ $preprocess = true ]; then
  echo "bla"
else
  echo "bloo"
fi

