#!/bin/bash -x
# This script sets up development and data environments for 
# a local machine, copy under your home directory and run.
# Note that, Theano is NOT installed by this script.

# code directory for cloned repositories
CODE_DIR=${HOME}/codes/dl4mt-material

# code repository 
CODE_CENTRAL=https://github.com/kyunghyuncho/dl4mt-material

# our input files will reside here
DATA_DIR=${HOME}/data

# our trained models will be saved here
MODELS_DIR=${HOME}/models


# clone the repository from github into code directory
if [ ! -d "${CODE_DIR}" ]; then
    mkdir -p ${CODE_DIR}
    git clone ${CODE_CENTRAL} ${CODE_DIR}
fi

# download the europarl v7 and validation sets and extract
python ${CODE_DIR}/data/download_files.py \
    -s='fr' -t='en' \
    --source-dev=newstest2011.fr \
    --target-dev=newstest2011.en \
    --outdir=${DATA_DIR}

# tokenize corresponding files
perl ${CODE_DIR}/data/tokenizer.perl -l 'fr' < ${DATA_DIR}/test2011/newstest2011.fr > ${DATA_DIR}/newstest2011.fr.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'en' < ${DATA_DIR}/test2011/newstest2011.en > ${DATA_DIR}/newstest2011.en.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'fr' < ${DATA_DIR}/europarl-v7.fr-en.fr > ${DATA_DIR}/europarl-v7.fr-en.fr.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'en' < ${DATA_DIR}/europarl-v7.fr-en.en > ${DATA_DIR}/europarl-v7.fr-en.en.tok

# extract dictionaries
python ${CODE_DIR}/data/build_dictionary.py ${DATA_DIR}/europarl-v7.fr-en.fr.tok
python ${CODE_DIR}/data/build_dictionary.py ${DATA_DIR}/europarl-v7.fr-en.en.tok

# create model output directory if it does not exist 
if [ ! -d "${MODELS_DIR}" ]; then
    mkdir -p ${MODELS_DIR}
fi

# check if theano is working
python -c "import theano;print 'theano available!'"
