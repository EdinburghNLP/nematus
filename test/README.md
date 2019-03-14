Testing Nematus
---------------

To test translation (on GPU 0), execute

CUDA_VISIBLE_DEVICES=0 python3 test_translate.py

To test scoring (on GPU 0), execute

CUDA_VISIBLE_DEVICES=0 python3 test_score.py

more sample models (including scripts for pre- and postprocessing)
are provided at: http://statmt.org/rsennrich/wmt16_systems/

to test training (on GPU 0), execute

CUDA_VISIBLE_DEVICES=0 ./test_train.sh

note that the training script is just a toy setup to make sure the scripts run,
and to allow for speed comparisons. For instructions to train a
real-scale system, check the instructions at https://github.com/rsennrich/wmt16-scripts
