NEMATUS
-------

Attention-based encoder-decoder model for neural machine translation

TUNEOUT BRANCH
--------------

This branch contains the implementation of the Tuneout fine-tuning technique from the paper:
Antonio Valerio Miceli Barone, Barry Haddow, Ulrich Germann, Rico Sennrich "Regularization techniques for fine-tuning in neural machine translation" 2017.
https://arxiv.org/abs/1707.09920


This package is based on the dl4mt-tutorial by Kyunghyun Cho et al. ( https://github.com/nyu-dl/dl4mt-tutorial ).
It was used to produce top-scoring systems at the WMT 16 shared translation task.

The changes to Nematus include:

 - arbitrary input features (factored neural machine translation) http://www.statmt.org/wmt16/pdf/W16-2209.pdf
 - ensemble decoding (and new translation API to support it)
 - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
 - minimum risk training (Shen et al, 2016) http://aclweb.org/anthology/P16-1159
 - command line interface for training
 - automatic training set reshuffling between epochs
 - n-best output for decoder
 - more output options (attention weights; word-level probabilities) and visualization scripts
 - performance improvements to decoder
 - rescoring support
 - execute arbitrary validation scripts (for BLEU early stopping)
 - vocabulary files and model parameters are stored in JSON format (backward-compatible loading)


INSTALLATION
------------

Nematus requires the following packages:

 - Python >= 2.7
 - numpy
 - ipdb
 - Theano >= 0.7 (and its dependencies).

we recommend executing the following command in a Python virtual environment:
   `pip install numpy numexpr cython tables theano ipdb`

the following packages are optional, but *highly* recommended

 - CUDA >= 7  (only GPU training is sufficiently fast)
 - cuDNN >= 3 (speeds up training substantially)


you can run Nematus locally. To install it, execute `python setup.py install`

DOCKER USAGE
------------

You can also create docker image by running following command, where you change `suffix` to either `cpu` or `gpu`:

`docker build -t nematus-docker -f Dockerfile.suffix .`

To run a CPU docker instance with the current working directory shared with the Docker container, execute:

``docker run -v `pwd`:/playground -it nematus-docker``

For GPU you need to have nvidia-docker installed and run:

``nvidia-docker run -v `pwd`:/playground -it nematus-docker``

USAGE INSTRUCTIONS
------------------

instructions to train a model are provided in https://github.com/rsennrich/wmt16-scripts

sample models, and instructions on using them for translation, are provided at http://statmt.org/rsennrich/wmt16_systems/


PUBLICATIONS
------------

the code is based on the following model:

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015): Neural Machine Translation by Jointly Learning to Align and Translate, Proceedings of the International Conference on Learning Representations (ICLR).

for the changes specific to Nematus, please consider the following papers:

Sennrich, Rico, Haddow, Barry, Birch, Alexandra (2016): Edinburgh Neural Machine Translation Systems for WMT 16, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany

Sennrich, Rico, Haddow, Barry (2016): Linguistic Input Features Improve Neural Machine Translation, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany
