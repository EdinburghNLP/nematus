Testing Nematus
---------------

To test translation, execute

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu python test_translate.py

To test scoring, execute

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu python test_score.py

more sample models (including scripts for pre- and postprocessing)
are provided at: http://statmt.org/rsennrich/wmt16_systems/

to test training, execute

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu ./test_train.sh

note that the training script is just a toy setup to make sure the scripts run,
and to allow for speed comparisons. For instructions to train a
real-scale system, check the instructions at https://github.com/rsennrich/wmt16-scripts