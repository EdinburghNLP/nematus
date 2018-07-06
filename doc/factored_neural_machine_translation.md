FACTORED NEURAL MACHINE TRANSLATION
-----------------------------------

Nematus supports arbitrary input features through factored representations, similar to factored models popularized with Moses.
This can be used to add linguistic features such as lemmas, POS, or dependency labels, or potentially other types of information.
The pipe symbol "|" serves as a factor separator and should not otherwise appear in the text.

To use factored models, follow these steps:

  - preprocess the source side of the training, development and test data to include factors. Consider this example sentence, in an unfactored (or 1-factored) representation, and with 4 factors per word:

    Leonidas begged in the arena .

    Leonidas|Leonidas|NNP|nsubj begged|beg|VBD|root in|in|IN|prep the|the|DT|det gladiatorial|gladiatorial|JJ|amod arena|arena|NN|pobj

    https://github.com/rsennrich/wmt16-scripts/tree/master/factored_sample provides sample scripts to produce a factored representation from a CoNLL file, and BPE-segmented text.

  - in the arguments to nematus.nmt.train, adjust the following options:
    - factors: the number of factors per word
    - dim_per_factor: the size of the embedding layer for each factor (a list of integers)
    - dim_word: the total size of the input embedding (must match the sum of dim_per_factor)
    - dictionaries: add a vocabulary file for each factor (in the order they appear), plus a vocabulary file for the target side

    an example config is shown at https://github.com/rsennrich/wmt16-scripts/blob/master/factored_sample/config.py

  - commands for training and running Nematus are otherwise identical to the non-factored version


PUBLICATIONS
------------

factored neural machine translation is described in:

Sennrich, Rico, Haddow, Barry (2016): Linguistic Input Features Improve Neural Machine Translation, Proc. of the First Conference on Machine Translation (WMT16). Berlin, Germany