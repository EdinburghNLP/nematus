#!/bin/bash -x
# This script sets up development and data environments for 
# fionn cluster, copy under your home directory and run.

# this file is for the dependencies
LOCAL_INSTALL_FILE=/ichec/work/dl4mt_data/local_install.tgz

# code directory for cloned repositories
CODE_DIR=${HOME}/codes/dl4mt-material

# code repository 
CODE_CENTRAL=https://github.com/kyunghyuncho/dl4mt-material

# reference files directory
REF_DATA_DIR=/ichec/work/dl4mt_data/nec_files

# our input files will reside here
DATA_DIR=${HOME}/data

# our trained models will be saved here
MODELS_DIR=${HOME}/models


# copy dependency file to your local and extract 
rsync --bwlimit=20000 -Pavz ${LOCAL_INSTALL_FILE} ${HOME}
tar zxvf ${HOME}/local_install.tgz

# clone the repository from github into code directory
if [ ! -d "${CODE_DIR}" ]; then
    mkdir -p ${CODE_DIR}
fi
git clone ${CODE_CENTRAL} ${CODE_DIR}

# copy corpora, dictionaries etc for training and dev
if [ ! -d "${DATA_DIR}" ]; then
    mkdir -p ${DATA_DIR}
fi
rsync --bwlimit=20000 -Pavz ${REF_DATA_DIR}/all.* ${DATA_DIR}
rsync --bwlimit=20000 -Pavz ${REF_DATA_DIR}/news* ${DATA_DIR}

# create model output directory if it does not exist 
if [ ! -d "${MODELS_DIR}" ]; then
    mkdir -p ${MODELS_DIR}
fi

# check if theano is working
python -c "import theano;print 'theano available!'"

