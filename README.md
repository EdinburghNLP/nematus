NEMATUS
-------

Attention-based encoder-decoder model for neural machine translation

This package is based on the dl4mt-tutorial by Kyunghyun Cho et al. ( https://github.com/nyu-dl/dl4mt-tutorial ).
It was used to produce top-scoring systems at the WMT 16 shared translation task.

The changes to Nematus include:

 - command line interface for training
 - automatic training set reshuffling between epochs
 - n-best output for decoder
 - performance improvements to decoder
 - rescoring support
 - vocabulary files and model parameters are stored in JSON format (backward-compatible loading)


SUPPORT
-------

For general support requests, there is a Google Groups mailing list at https://groups.google.com/d/forum/nematus-support . You can also send an e-mail to nematus-support@googlegroups.com .

INSTALLATION
------------

Nematus requires the following packages:

 - Python >= 2.7
 - tensorflow

To install tensorflow, we recommend following the steps at:
  ( https://www.tensorflow.org/install/ )

the following packages are optional, but *highly* recommended

 - CUDA >= 7  (only GPU training is sufficiently fast)
 - cuDNN >= 4 (speeds up training substantially)

USAGE INSTRUCTIONS
------------------

All of the scripts below can be run with `--help` flag to get usage information.

#### `nematus/nmt.py` : use to train a new model

#### `nematus/translate.py` : use an existing model to translate a source text

#### `nematus/score.py` : use an existing model to score a parallel corpus

#### `nematus/rescore.py` : use an existing model to rescore an n-best list.

The n-best list is assumed to have the same format as Moses:

    sentence-ID (starting from 0) ||| translation ||| scores

new scores will be appended to the end. 

#### `nematus/theano_tf_convert.py` : convert an existing theano model to a tensorflow model

If you have a Theano model (model.npz) with network architecture features that are currently
supported then you can convert it into a tensorflow model using `nematus/theano_tf_convert.py`.


PUBLICATIONS
------------

the code is based on the following model:

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015): Neural Machine Translation by Jointly Learning to Align and Translate, Proceedings of the International Conference on Learning Representations (ICLR).

for the changes specific to Nematus, please consider the following papers:

Sennrich, Rico, Haddow, Barry, Birch, Alexandra (2016): Edinburgh Neural Machine Translation Systems for WMT 16, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany

Sennrich, Rico, Haddow, Barry (2016): Linguistic Input Features Improve Neural Machine Translation, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany
