#!/bin/bash
set -e
scripts_dir=`pwd`
source ${scripts_dir}/consts.sh
#############cleanup###############
echo "###############cleanup###############"
python ${nematus_dir}/cleanup.py
sh run_all_flows.sh de 0
sh run_all_flows.sh de 1
sh run_all_flows.sh he 0
sh run_all_flows.sh he 1
sh run_all_flows.sh ru 0
sh run_all_flows.sh ru 1
