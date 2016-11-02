NEMATUS
-------

Attention-based encoder-decoder model for neural machine translation

This package is based on the dl4mt-tutorial by Kyunghyun Cho et al. ( https://github.com/nyu-dl/dl4mt-tutorial ).
It was used to produce top-scoring systems at the WMT 16 shared translation task.

The changes to Nematus include:

 - arbitrary input features (factored neural machine translation) http://www.statmt.org/wmt16/pdf/W16-2209.pdf
 - ensemble decoding (and new translation API to support it)
 - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
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
 - cuDNN >= 4 (speeds up training substantially)


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

### TRAINING

execute nematus/nmt.py to train a model.


#### data sets; model loading and saving
| parameter            | description |
|---                   |--- |
| --datasets PATH PATH |  parallel training corpus (source and target) |
| --dictionaries PATH [PATH ...] | network vocabularies (one per source factor, plus target vocabulary) |
| --model PATH         |  model file name (default: model.npz) |
| --saveFreq INT       |  save frequency (default: 30000) |
| --reload_            |  load existing model (if '--model' points to existing model) |
| --overwrite          |  write all models to same file |

#### network parameters
| parameter            | description |
|---                   |--- |
| --dim_word INT       |  embedding layer size (default: 512) |
| --dim INT            |  hidden layer size (default: 1000) |
| --n_words_src INT    |  source vocabulary size (default: None) |
| --n_words INT        |  target vocabulary size (default: None) |
| --factors INT        |  number of input factors (default: 1) |
| --dim_per_factor INT [INT ...] | list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: None) |
| --use_dropout        |  use dropout layer (default: False) |
| --dropout_embedding FLOAT | dropout for input embeddings (0: no dropout) (default: 0.2) |
| --dropout_hidden FLOAT | dropout for hidden layer (0: no dropout) (default: 0.2) |
| --dropout_source FLOAT | dropout source words (0: no dropout) (default: 0) |
| --dropout_target FLOAT | dropout target words (0: no dropout) (default: 0) |

#### training parameters
| parameter            | description |
|---                   |--- |
| --maxlen INT         |  maximum sequence length (default: 100) |
| --optimizer {adam,adadelta,rmsprop,sgd} | optimizer (default: adam) |
| --batch_size INT     | minibatch size (default: 80) |
| --max_epochs INT     | maximum number of epochs (default: 5000) |
| --finish_after INT   | maximum number of updates (minibatches) (default: 10000000) |
| --decay_c FLOAT      |  L2 regularization penalty (default: 0) |
| --map_decay_c FLOAT  |  L2 regularization penalty towards original weights (default: 0) |
| --alpha_c FLOAT      |  alignment regularization (default: 0) |
| --clip_c FLOAT       |  gradient clipping threshold (default: 1) |
| --lrate FLOAT        |  learning rate (default: 0.0001) |
| --no_shuffle         |  disable shuffling of training data (for each epoch) |
| --no_sort_by_length  |  do not sort sentences in maxibatch by length |
| --maxibatch_size INT |  size of maxibatch (number of minibatches that are sorted by length) (default: 20) |
| --finetune           |  train with fixed embedding layer |
| --finetune_only_last |  train with all layers except output layer fixed |

#### validation parameters
| parameter            | description |
|---                   |--- |
| --valid_datasets PATH PATH | parallel validation corpus (source and target)| (default: None) |
| --valid_batch_size INT | validation minibatch size (default: 80) |
| --validFreq INT       | validation frequency (default: 10000) |
| --patience INT        | early stopping patience (default: 10) |
| --external_validation_script PATH | location of validation script (to run your favorite metric for validation) (default: None) |

#### display parameters
| parameter            | description |
|---                   |--- |
| --dispFreq INT       | display loss after INT updates (default: 1000) |
| --sampleFreq INT     | display some samples after INT updates (default: 10000) |



more instructions to train a model, including a sample configuration and
preprocessing scripts, are provided in https://github.com/rsennrich/wmt16-scripts

### USING A TRAINED MODEL

#### `nematus/translate.py` : use an existing model to translate a source text

| parameter            | description |
|---                   |--- |
| -k K                 | Beam size (default: 5)) |
|-p P                  | Number of processes (default: 5)) |
| -n                   | Normalize scores by sentence length |
| -v                   | verbose mode. |
| --models MODELS [MODELS ...], -m MODELS [MODELS ...] | model to use. Provide multiple models (with same vocabulary) for ensemble decoding |
|--input PATH, -i PATH | Input file (default: standard input) |
| --output PATH, -o PATH | Output file (default: standard output) |
| --output_alignment PATH, -a PATH | Output file for alignment weights (default: standard output) |
| --json_alignment     | Output alignment in json format |
| --n-best             | Write n-best list (of size k) |
| --suppress-unk       | Suppress hypotheses containing UNK. |
| --print-word-probabilities, -wp | Print probabilities of each word |


#### `nematus/score.py` : use an existing model to score a parallel corpus

| parameter              | description |
|---                     |--- |
| -b B                   |   Minibatch size (default: 80)) |
| -n                     |   Normalize scores by sentence length |
| -v                     |   verbose mode. |
| --models MODELS [MODELS ...], -m MODELS [MODELS ...] | model to use. Provide multiple models (with same vocabulary) for ensemble decoding |
| --source PATH, -s PATH | Source text file |
| --target PATH, -t PATH | Target text file |
| --output PATH, -o PATH | Output file (default: standard output) |
| --walign, -w           | Whether to store the alignment weights or not. If specified, weights will be saved in <target>.alignment |


#### `nematus/rescore.py` : use an existing model to rescore an n-best list.

The n-best list is assumed to have the same format as Moses:

    sentence-ID (starting from 0) ||| translation ||| scores

new scores will be appended to the end. `rescore.py` has the same arguments as `score.py`, with the exception of this additional parameter:

| parameter             | description |
|---                    |--- |
| --input PATH, -i PATH | Input n-best list file (default: standard input) |


sample models, and instructions on using them for translation, are provided in the `test` directory, and at http://statmt.org/rsennrich/wmt16_systems/

PUBLICATIONS
------------

the code is based on the following model:

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015): Neural Machine Translation by Jointly Learning to Align and Translate, Proceedings of the International Conference on Learning Representations (ICLR).

for the changes specific to Nematus, please consider the following papers:

Sennrich, Rico, Haddow, Barry, Birch, Alexandra (2016): Edinburgh Neural Machine Translation Systems for WMT 16, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany

Sennrich, Rico, Haddow, Barry (2016): Linguistic Input Features Improve Neural Machine Translation, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany
