DEEP FUSION OF LANGUAGE MODEL WITH MT DECODER
---------------------------------------------

Deep fusion models combines a RNN language model (LM)
with the MT decoder. The language model is trained
in advance and its parameters are fixed during the
training of the MT model.

Train a LM using DL4MT: https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session0

The vocabulary of the LM is the same as the target
vocabulary of the MT model. Thus obtain first the
target vocabulary dictionary of the translation
model (`data/build_dictionary.py`). Then, convert
this dictionary to DL4MT format (`data/dict_nematus_to_dl4mt.py`).

Once the LM is trained, we can start the deep fusion
model training. Here are the relevant arguments in Nematus:

- `deep_fusion_lm`: path to the LM npz file. The system will also load the pickle file containing the LM options, which should be in the same directory.
- `concatenate_lm_decoder` [optional]: by default, the LM and decoder states are linearly projected and summed. Add this argument for a concatenation of both states instead.

PUBLICATIONS
------------

The deep fusion model in Nematus is based on the following paper:

Çaglar Gülçehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Loïc Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2015): On Using Monolingual Corpora in Neural Machine Translation, CoRR, http://arxiv.org/abs/1503.03535

