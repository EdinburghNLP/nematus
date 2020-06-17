NEMATUS
-------

Attention-based encoder-decoder model for neural machine translation built in Tensorflow.

Notable features include:

  - support for RNN and Transformer architectures

  - support for advanced RNN architectures:
     - [arbitrary input features](doc/factored_neural_machine_translation.md) (factored neural machine translation) http://www.statmt.org/wmt16/pdf/W16-2209.pdf
     - deep models (Miceli Barone et al., 2017) https://arxiv.org/abs/1707.07631
     - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
     - tied embeddings (Press and Wolf, 2016) https://arxiv.org/abs/1608.05859
     - layer normalisation (Ba et al, 2016) https://arxiv.org/abs/1607.06450
     - mixture of softmaxes (Yang et al., 2017) https://arxiv.org/abs/1711.03953
     - lexical model (Nguyen and Chiang, 2018) https://www.aclweb.org/anthology/N18-1031

  - support for advanced Transformer architectures:
     - DropHead: dropout of entire attention heads (Zhou et al., 2020) https://arxiv.org/abs/2004.13342

 - training features:
     - multi-GPU support [documentation](doc/multi_gpu_training.md)
     - label smoothing
     - early stopping with user-defined stopping criterion
     - resume training (optionally with MAP-L2 regularization towards original model)
     - minimum risk training (MRT)

 - scoring and decoding features:
     - batch decoding
     - n-best output
     - scripts for scoring (given parallel corpus) and rescoring (of n-best output)
     - server mode

 - other usability features:
     - command line interface for training, scoring, and decoding
     - JSON-formatted storage of model hyperparameters, vocabulary files and training progress
     - pretrained models for 13 translation directions (many top-performing at WMT shared task of respective year):
       - http://data.statmt.org/rsennrich/wmt16_systems/
       - http://data.statmt.org/wmt17_systems/
     - backward compatibility: continue using publicly released models with current codebase (scripts to convert from Theano to Tensorflow-style models are provided)


SUPPORT
-------

For general support requests, there is a Google Groups mailing list at https://groups.google.com/d/forum/nematus-support . You can also send an e-mail to nematus-support@googlegroups.com .


INSTALLATION
------------

Nematus requires the following packages:

 - Python 3 (tested on version 3.5.2)
 - TensorFlow 1.15 / 2.X (tested on version 2.0)

To install tensorflow, we recommend following the steps at:
  ( https://www.tensorflow.org/install/ )

the following packages are optional, but *highly* recommended

 - CUDA >= 7  (only GPU training is sufficiently fast)
 - cuDNN >= 4 (speeds up training substantially)


LEGACY THEANO VERSION
---------------------

Nematus originated as a fork of dl4mt-tutorial by Kyunghyun Cho et al. ( https://github.com/nyu-dl/dl4mt-tutorial ), and was implemented in Theano.
See https://github.com/EdinburghNLP/nematus/tree/theano for this Theano-based version of Nematus.

To use models trained with Theano with the current Tensorflow codebase, use the script `nematus/theano_tf_convert.py`.

DOCKER USAGE
------------

You can also create docker image by running following command, where you change `suffix` to either `cpu` or `gpu`:

`docker build -t nematus-docker -f Dockerfile.suffix .`

To run a CPU docker instance with the current working directory shared with the Docker container, execute:

``docker run -v `pwd`:/playground -it nematus-docker``

For GPU you need to have nvidia-docker installed and run:

``nvidia-docker run -v `pwd`:/playground -it nematus-docker``


TRAINING SPEED
--------------

Training speed depends heavily on having appropriate hardware (ideally a recent NVIDIA GPU),
and having installed the appropriate software packages.

To test your setup, we provide some speed benchmarks with `test/test_train.sh',
on an Intel Xeon CPU E5-2620 v4, with a Nvidia GeForce GTX Titan X (Pascal) and CUDA 9.0:


GPU, CuDNN 5.1, tensorflow 1.0.1:

  CUDA_VISIBLE_DEVICES=0 ./test_train.sh

>> 225.25 sentenses/s

 
USAGE INSTRUCTIONS
------------------

All of the scripts below can be run with `--help` flag to get usage information.

Sample commands with toy examples are available in the `test` directory;
for training a full-scale RNN system, consider the training scripts at http://data.statmt.org/wmt17_systems/training/

An updated version of these scripts that uses the Transformer model can be found at https://github.com/EdinburghNLP/wmt17-transformer-scripts

#### `nematus/train.py` : use to train a new model

#### data sets; model loading and saving
| parameter | description |
|---        |---          |
| --source_dataset PATH | parallel training corpus (source) |
| --target_dataset PATH | parallel training corpus (target) |
| --dictionaries PATH [PATH ...] | network vocabularies (one per source factor, plus target vocabulary) |
| --save_freq INT | save frequency (default: 30000) |
| --model PATH | model file name (default: model) |
| --reload PATH | load existing model from this path. Set to "latest_checkpoint" to reload the latest checkpoint in the same directory of --model |
| --no_reload_training_progress | don't reload training progress (only used if --reload is enabled) |
| --summary_dir PATH | directory for saving summaries (default: same directory as the --model file) |
| --summary_freq INT | Save summaries after INT updates, if 0 do not save summaries (default: 0) |

#### network parameters (all model types)
| parameter | description |
|---        |---          |
| --model_type {rnn,transformer} | model type (default: rnn) |
| --embedding_size INT | embedding layer size (default: 512) |
| --state_size INT | hidden state size (default: 1000) |
| --source_vocab_sizes INT [INT ...] | source vocabulary sizes (one per input factor) (default: None) |
| --target_vocab_size INT | target vocabulary size (default: -1) |
| --factors INT | number of input factors (default: 1) - CURRENTLY ONLY WORKS FOR 'rnn' MODEL |
| --dim_per_factor INT [INT ...] | list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: None) |
| --tie_encoder_decoder_embeddings | tie the input embeddings of the encoder and the decoder (first factor only). Source and target vocabulary size must be the same |
| --tie_decoder_embeddings | tie the input embeddings of the decoder with the softmax output embeddings |
| --output_hidden_activation {tanh,relu,prelu,linear} | activation function in hidden layer of the output network (default: tanh) - CURRENTLY ONLY WORKS FOR 'rnn' MODEL |
| --softmax_mixture_size INT | number of softmax components to use (default: 1) - CURRENTLY ONLY WORKS FOR 'rnn' MODEL |

#### network parameters (rnn-specific)
| parameter | description |
|---        |---          |
| --rnn_enc_depth INT | number of encoder layers (default: 1) |
| --rnn_enc_transition_depth INT | number of GRU transition operations applied in the encoder. Minimum is 1. (Only applies to gru). (default: 1) |
| --rnn_dec_depth INT | number of decoder layers (default: 1) |
| --rnn_dec_base_transition_depth INT | number of GRU transition operations applied in the first layer of the decoder. Minimum is 2. (Only applies to gru_cond). (default: 2) |
| --rnn_dec_high_transition_depth INT | number of GRU transition operations applied in the higher layers of the decoder. Minimum is 1. (Only applies to gru). (default: 1) |
| --rnn_dec_deep_context | pass context vector (from first layer) to deep decoder layers |
| --rnn_dropout_embedding FLOAT | dropout for input embeddings (0: no dropout) (default: 0.0) |
| --rnn_dropout_hidden FLOAT | dropout for hidden layer (0: no dropout) (default: 0.0) |
| --rnn_dropout_source FLOAT | dropout source words (0: no dropout) (default: 0.0) |
| --rnn_dropout_target FLOAT | dropout target words (0: no dropout) (default: 0.0) |
| --rnn_layer_normalisation | Set to use layer normalization in encoder and decoder |
| --rnn_lexical_model | Enable feedforward lexical model (Nguyen and Chiang, 2018) |

#### network parameters (transformer-specific)
| parameter | description |
|---        |---          |
| --transformer_enc_depth INT | number of encoder layers (default: 6) |
| --transformer_dec_depth INT | number of decoder layers (default: 6) |
| --transformer_ffn_hidden_size INT | inner dimensionality of feed-forward sub-layers (default: 2048) |
| --transformer_num_heads INT | number of attention heads used in multi-head attention (default: 8) |
| --transformer_dropout_embeddings FLOAT | dropout applied to sums of word embeddings and positional encodings (default: 0.1) |
| --transformer_dropout_residual FLOAT | dropout applied to residual connections (default: 0.1) |
| --transformer_dropout_relu FLOAT | dropout applied to the internal activation of the feed-forward sub-layers (default: 0.1) |
| --transformer_dropout_attn FLOAT | dropout applied to attention weights (default: 0.1) |
| --transformer_drophead FLOAT | dropout of entire attention heads (default: 0.0) |

#### training parameters
| parameter | description |
|---        |---          |
| --loss_function {cross-entropy,per-token-cross-entropy, MRT} | loss function. MRT: Minimum Risk Training https://www.aclweb.org/anthology/P/P16/P16-1159.pdf) (default: cross-entropy) |
| --decay_c FLOAT | L2 regularization penalty (default: 0.0) |
| --map_decay_c FLOAT | MAP-L2 regularization penalty towards original weights (default: 0.0) |
| --prior_model PATH | Prior model for MAP-L2 regularization. Unless using " --reload", this will also be used for initialization. |
| --clip_c FLOAT | gradient clipping threshold (default: 1.0) |
| --label_smoothing FLOAT | label smoothing (default: 0.0) |
| --exponential_smoothing FLOAT | exponential smoothing factor; use 0 to disable (default: 0.0) |
| --optimizer {adam} | optimizer (default: adam) |
| --adam_beta1 FLOAT | exponential decay rate for the first moment estimates (default: 0.9) |
| --adam_beta2 FLOAT | exponential decay rate for the second moment estimates (default: 0.999) |
| --adam_epsilon FLOAT | constant for numerical stability (default: 1e-08) |
| --learning_schedule {constant,transformer,warmup-plateau-decay} | learning schedule (default: constant) |
| --learning_rate FLOAT | learning rate (default: 0.0001) |
| --warmup_steps INT | number of initial updates during which the learning rate is increased linearly during learning rate scheduling (default: 8000) |
| --plateau_steps INT | number of updates after warm-up before the learning rate starts to decay (applies to 'warmup-plateau-decay' learning schedule only). (default: 0) |
| --maxlen INT | maximum sequence length for training and validation (default: 100) |
| --batch_size INT | minibatch size (default: 80) |
| --token_batch_size INT | minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, batch_size only affects sorting by length. (default: 0) |
| --max_sentences_per_device INT | maximum size of minibatch subset to run on a single device, in number of sentences (default: 0) |
| --max_tokens_per_device INT | maximum size of minibatch subset to run on a single device, in number of tokens (either source or target - whichever is highest) (default: 0) |
| --gradient_aggregation_steps INT | number of times to accumulate gradients before aggregating and applying; the minibatch is split between steps, so adding more steps allows larger minibatches to be used (default: 1) |
| --maxibatch_size INT | size of maxibatch (number of minibatches that are sorted by length) (default: 20) |
| --no_sort_by_length | do not sort sentences in maxibatch by length |
| --no_shuffle | disable shuffling of training data (for each epoch) |
| --keep_train_set_in_memory | Keep training dataset lines stores in RAM during training |
| --max_epochs INT | maximum number of epochs (default: 5000) |
| --finish_after INT | maximum number of updates (minibatches) (default: 10000000) |
| --print_per_token_pro PATH | PATH to store the probability of each target token given source sentences over the training dataset (without training). If set to False, the function will not be triggered. (default: False). Please get rid of the 1.0s at the end of each list which are the probability of padding. |

#### minimum risk training parameters (MRT)

| parameter | description |
|---        |---          |
| --mrt_reference | add reference into MRT candidates sentences (default: False) |
| --mrt_alpha FLOAT | MRT alpha to control the sharpness of the distribution of sampled subspace (default: 0.005) |
| --samplesN INT | the number of sampled candidates sentences per source sentence (default: 100) |
| --mrt_loss | evaluation metrics used to compute loss between the candidate translation and reference translation (default: SENTENCEBLEU n=4) |
| --mrt_ml_mix FLOAT | mix in MLE objective in MRT training with this scaling factor (default: 0) |
| --sample_way {beam_search, randomly_sample} | the sampling strategy to generate candidates sentences (default: beam_search) |
| --max_len_a INT | generate candidates sentences with maximum length: ax + b, where x is the length of the source sentence (default: 1.5) |
| --max_len_b INT | generate candidates sentences with maximum length: ax + b, where x is the length of the source sentence (default: 5) |
| --max_sentences_of_sampling INT | maximum number of source sentences to generate candidates sentences at one time (limited by device memory capacity) (default: 0) |

#### validation parameters
| parameter | description |
|---        |---          |
| --valid_source_dataset PATH | source validation corpus (default: None) |
| --valid_target_dataset PATH | target validation corpus (default: None) |
| --valid_batch_size INT | validation minibatch size (default: 80) |
| --valid_token_batch_size INT | validation minibatch size (expressed in number of source or target tokens). Sentence-level minibatch size will be dynamic. If this is enabled, valid_batch_size only affects sorting by length. (default: 0) |
| --valid_freq INT | validation frequency (default: 10000) |
| --valid_script PATH | path to script for external validation (default: None). The script will be passed an argument specifying the path of a file that contains translations of the source validation corpus. It must write a single score to standard output. |
| --valid_bleu_source_dataset PATH | source validation corpus for external validation (default: None). If set to None, the dataset for calculating validation loss (valid_source_dataset) will be used |
| --patience INT | early stopping patience (default: 10) |

#### display parameters
| parameter | description |
|---        |---          |
| --disp_freq INT | display loss after INT updates (default: 1000) |
| --sample_freq INT | display some samples after INT updates (default: 10000) |
| --beam_freq INT | display some beam_search samples after INT updates (default: 10000) |
| --beam_size INT | size of the beam (default: 12) |

#### translate parameters
| parameter | description |
|---        |---          |
| --normalization_alpha [ALPHA] | normalize scores by sentence length (with argument, " "exponentiate lengths by ALPHA) |
| --n_best | Print full beam |
| --translation_maxlen INT | Maximum length of translation output sentence (default: 200) |
| --translation_strategy {beam_search,sampling} | translation_strategy, either beam_search or sampling (default: beam_search) |

#### `nematus/translate.py` : use an existing model to translate a source text

| parameter | description |
|---        |---          |
| -v, --verbose | verbose mode |
| -m PATH [PATH ...], --models PATH [PATH ...] | model to use; provide multiple models (with same vocabulary) for ensemble decoding |
| -b INT, --minibatch_size INT | minibatch size (default: 80) |
| -i PATH, --input PATH | input file (default: standard input) |
| -o PATH, --output PATH | output file (default: standard output) |
| -k INT, --beam_size INT | beam size (default: 5) |
| -n [ALPHA], --normalization_alpha [ALPHA] | normalize scores by sentence length (with argument, exponentiate lengths by ALPHA) |
| --n_best | write n-best list (of size k) |
| --maxibatch_size INT | size of maxibatch (number of minibatches that are sorted by length) (default: 20) |

#### `nematus/score.py` : use an existing model to score a parallel corpus

| parameter | description |
|---        |---          |
| -v, --verbose | verbose mode |
| -m PATH [PATH ...], --models PATH [PATH ...] | model to use; provide multiple models (with same vocabulary) for ensemble decoding |
| -b INT, --minibatch_size INT | minibatch size (default: 80) |
| -n [ALPHA], --normalization_alpha [ALPHA] | normalize scores by sentence length (with argument, exponentiate lengths by ALPHA) |
| -o PATH, --output PATH | output file (default: standard output) |
| -s PATH, --source PATH | source text file |
| -t PATH, --target PATH | target text file |


#### `nematus/rescore.py` : use an existing model to rescore an n-best list.

The n-best list is assumed to have the same format as Moses:

    sentence-ID (starting from 0) ||| translation ||| scores

new scores will be appended to the end. `rescore.py` has the same arguments as `score.py`, with the exception of this additional parameter:

| parameter             | description |
|---                    |--- |
| -i PATH, --input PATH | input n-best list file (default: standard input) |


#### `nematus/theano_tf_convert.py` : convert an existing theano model to a tensorflow model

If you have a Theano model (model.npz) with network architecture features that are currently
supported then you can convert it into a tensorflow model using `nematus/theano_tf_convert.py`.

| parameter | description |
|---        |---          |
| --from_theano | convert from Theano to TensorFlow format |
| --from_tf | convert from Tensorflow to Theano format |
| --in PATH | path to input model |
| --out PATH | path to output model |


PUBLICATIONS
------------

if you use Nematus, please cite the following paper:

Rico Sennrich, Orhan Firat, Kyunghyun Cho, Alexandra Birch, Barry Haddow, Julian Hitschler, Marcin Junczys-Dowmunt, Samuel Läubli, Antonio Valerio Miceli Barone, Jozef Mokry and Maria Nadejde (2017): Nematus: a Toolkit for Neural Machine Translation. In Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics, Valencia, Spain, pp. 65-68.

```
@InProceedings{sennrich-EtAl:2017:EACLDemo,
  author    = {Sennrich, Rico  and  Firat, Orhan  and  Cho, Kyunghyun  and  Birch, Alexandra  and  Haddow, Barry  and  Hitschler, Julian  and  Junczys-Dowmunt, Marcin  and  L\"{a}ubli, Samuel  and  Miceli Barone, Antonio Valerio  and  Mokry, Jozef  and  Nadejde, Maria},
  title     = {Nematus: a Toolkit for Neural Machine Translation},
  booktitle = {Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {65--68},
  url       = {http://aclweb.org/anthology/E17-3017}
}
```

the code is based on the following models:

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015): Neural Machine Translation by Jointly Learning to Align and Translate, Proceedings of the International Conference on Learning Representations (ICLR).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017): Attention is All You Need, Advances in Neural Information Processing Systems (NIPS).

please refer to the Nematus paper for a description of implementation differences to the RNN model.


ACKNOWLEDGMENTS
---------------
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreements 645452 (QT21), 644333 (TraMOOC), 644402 (HimL) and 688139 (SUMMA).
