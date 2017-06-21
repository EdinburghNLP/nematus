All these experiments use theano 0.9.0-RELEASE, CuDNN 5.1, new GPU backend.

TITAN X
-------

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32:

>> 232.70 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float16:

>> 232.30 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float64:

>> 66.39 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32,gpuarray.preallocate=0.8 --batch_size 256

>> 477.41 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float16,gpuarray.preallocate=0.8 --batch_size 256

>> 491.57 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float64,gpuarray.preallocate=0.8 --batch_size 256

>> out of memory

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32,gpuarray.preallocate=0.8 --batch_size 256 --dim_word 512 --dim 1024

>> 183.47 sentences/s

GPU TITAN X (Pascal), Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz, float32,gpuarray.preallocate=0.8 --batch_size 256 --dim_word 512 --dim 1024

>> NaN detected

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float32,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> 198.74

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float16,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> NaN detected

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float64,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> 25.35

Tesla P100-PCIE-16GB
--------------------

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

>> NaN detected

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float64,gpuarray.preallocate=0.8 --batch_size 256 --dim_word 512 --dim 1024

>> out of memory

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float32,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> 166.06 sentences/s

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float16,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> NaN detected

GPU Tesla P100-PCIE-16GB, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, new GPU backend,float64,gpuarray.preallocate=0.8 --batch_size 80 --dim_word 512 --dim 1024

>> 120.15 sentences/s
