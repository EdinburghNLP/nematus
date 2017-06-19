TRAINING SPEED
--------------

Training speed depends heavily on having appropriate hardware (ideally a recent NVIDIA GPU),
and having installed the appropriate software packages.

To test your setup, we provide some speed benchmarks with `test/test_train.sh',
on an Intel Xeon CPU E5-2620 v3, with a Nvidia GeForce GTX 1080 GPU and CUDA 8.0:

CPU, theano 0.8.2:

  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu ./test_train.sh

>> 2.37 sentences/s

GPU, no CuDNN, theano 0.8.2:

  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu ./test_train.sh

>> 71.62 sentences/s

GPU, CuDNN 5.1, theano 0.8.2:

  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu ./test_train.sh

>> 139.73 sentences/s

GPU, CuDNN 5.1, theano 0.9.0dev5.dev-d5520e:

  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu ./test_train.sh

>> 173.15 sentences/s

GPU, CuDNN 5.1, theano 0.9.0dev5.dev-d5520e, new GPU backend:

  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda ./test_train.sh

>> 209.21 sentences/s

GPU, float16, CuDNN 5.1, theano 0.9.0-RELEASE, new GPU backend:

>> 222.28 sentences/s

Other hardware
--------------

All these experiments use theano 0.9.0-RELEASE, CuDNN 5.1, new GPU backend.

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float16:

>> 232.30 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32:

>> 232.70 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float64:

>> 66.39 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32,gpuarray.preallocate=0.8 --batch_size 256

>> 477.41 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float16,gpuarray.preallocate=0.8 --batch_size 256

>> 491.57 sents/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float64,gpuarray.preallocate=0.8 --batch_size 256

>> out of memory

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend, float32:

>> 178.61 sentences/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend, float16:

>> 166.19 sentencess/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend, float64:

>> 162.78 sents/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float32,gpuarray.preallocate=0.8 --batch_size 256

>> 453.74 sentences/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float16,gpuarray.preallocate=0.8 --batch_size 256

>> 441.75 sentencs/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float64,gpuarray.preallocate=0.8 --batch_size 256

>> 325.12 sentences/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float32,gpuarray.preallocate=0.8 --batch_size 256 --dim_word 512 --dim 1024

>> 246.28 sentences/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float16,gpuarray.preallocate=0.8 --batch_size 256 --dim_word 512 --dim 1024

>> 

