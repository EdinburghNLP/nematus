# dl4mt-material
Attention-based encoder-decoder model for machine translatio

This package is based on dl4mt-material by Kyunghyun Cho ( https://github.com/kyunghyuncho/dl4mt-material )

Our changes include:

 - ensemble decoding
 - dropout on all layers (Gal, 2015) http://arxiv.org/abs/1512.05287
 - n-best output for decoder
 - performance improvements to decoder
 - rescoring



## Training
Change the hard-coded paths to data in `nmt.py` then run
```
THEANO_FLAGS=device=gpu,floatX=float32 python train_nmt.py 
```

