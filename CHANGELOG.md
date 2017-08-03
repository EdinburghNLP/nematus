CHANGELOG
---------

v0.2 (dev)
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
