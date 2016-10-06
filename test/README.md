Testing Nematus
---------------

To test training, execute

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu python test_translate.py

more sample models (including scripts for pre- and postprocessing)
are provided at: http://statmt.org/rsennrich/wmt16_systems/

to test training, a sample setup is provided in https://github.com/rsennrich/wmt16-scripts