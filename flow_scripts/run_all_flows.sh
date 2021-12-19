#!/bin/csh
set -e
echo $SHELL
#sh print_embedding_table.sh $1
#sh evaluate_gender_bias.sh $1
sh evaluate_translation.sh $1
