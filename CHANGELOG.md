CHANGELOG
---------

development version
-----------

v0.5 (19/5/2020)
----------

changes since 0.4:

 - main new features:
   - minimum risk training (MRT)
   - new inference code with ensemble decoding support for Transformer/RNN mix
   - compatibility with TF 2

 - other new features
   - lexical model for RNNs
   - gradient accumulation support
   - exponential smoothing
   - warmup-plateau-decay learning schedule
   - sampling translation strategy

 - fixes
   - fix regressions with deep RNN decoders


v0.4 (17/12/2018)
----------

changes since 0.3:

 - main new features:
   - Transformer architecture
   - multi-GPU training
   - codebase moved to Python 3

 - other new features:
   - label smoothing
   - mixture of softmaxes

 - fixes:
   - re-enable BLEU validation (via --valid_script)
   - fix MAP-L2 regularization
   - fix server mode

v0.3 (23/5/2018)
----------
 - Tensorflow backend. The main model was rewritten to support Tensorflow in lieu of Theano.
   A few features have not been implemented in the Tensorflow model.

 - currently supported:
   - re-implementation of default Nematus model
   - model compatibility with Theano version and conversion via `theano_tf_convert.py`
   - same scripts and command line API for training, translating and (re)scoring
   - layer normalisation
   - tied embeddings
   - deep models
   - ensemble decoding
   - input features
 
 - not yet supported:
   - minimum risk training
   - LSTM cells
   - learning rate annealing

 - new features:
   - batch decoding
   - more efficient training with --token_batch_size

v0.2 (17/12/2017)
----------

 - layer normalisation (Ba et al, 2016) https://arxiv.org/abs/1607.06450
 - weight normalisation (Salimans and Kingma, 2016) https://arxiv.org/abs/1602.07868
 - deep models (Zhou et al., 2016; Wu et al., 2016; Miceli Barone et al., 2017) https://arxiv.org/abs/1606.04199 https://arxiv.org/abs/1609.08144 https://arxiv.org/abs/1707.07631
 - better memory efficiency
 - save historical gradient information for seamless resuming of interrupted training runs
 - server mode
 - sgdmomentum optimizer
 - learning rate annealing
 - LSTM cells
 - deep fusion (https://arxiv.org/abs/1503.03535)
 - various bugfixes

v0.1 (2/3/2017)
---------------

 - arbitrary input features (factored neural machine translation) http://www.statmt.org/wmt16/pdf/W16-2209.pdf
 - ensemble decoding (and new translation API to support it)
 - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
 - minimum risk training (Shen et al, 2016) http://aclweb.org/anthology/P16-1159
 - tied embeddings (Press and Wolf, 2016) https://arxiv.org/abs/1608.05859
 - command line interface for training
 - n-best output for decoder
 - more output options (attention weights; word-level probabilities) and visualization scripts
 - performance improvements to decoder
 - better memory efficiency
 - rescoring support
 - execute arbitrary validation scripts (for BLEU early stopping)
 - vocabulary files and model parameters are stored in JSON format (backward-compatible loading)
