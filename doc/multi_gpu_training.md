Multi-GPU Training with Nematus
-------------------------------

Nematus supports multi-GPU training; this shows how to make the best use of it.

Controlling devices:
--------------------

by default, Nematus will split training across all available devices.
To control which device(s) to use for training, use `CUDA_VISIBLE_DEVICES`.

For example, this command uses the first two devices:

```
CUDA_VISIBLE_DEVICES=0,1 python3 nematus/train.py
```

Update strategy and batch size:
-------------------------------

Nematus will perform an update after a fixed number of sentences (`--batch_size`) or tokens (rounded down to full sentences; `--token_batch_size`). If both are defined, `--token_batch_size` takes priority.

When training on multiple devices, Nematus uses Synchronous SGD, and sentences in a batch are split between GPUs.
We choose this strategy for transparency. In principle, if training a model on the same data with the same command line parameters,
you should get similar results (except for random variation), even if systems are trained on different (number of) GPUs.

Generally, you should choose a large batch size to benefit from multi-GPU training and stabilize training of Transformers.
Our [baseline configuration](https://github.com/EdinburghNLP/wmt17-transformer-scripts/blob/master/training/scripts/train.sh) uses a `token_batch_size` of 16384,
and was tested on 4 GPUs with 12GB of memory each.

If you want to train a model with a batch size between updates that exceeds the memory available on your devices (because you are limited in the size and/or number of GPUs),
Nematus supports two ways of further splitting up the batch.

 - define `--max_sentences_per_device` or `--max_tokens_per_device`. This is the size of the batch that is processed on a single device at once. Batches are accumulated until reaching the total batch size. For example, defining `--max_tokens_per_device 4096` should ensure that the Transformer baseline will train successfully with 1-4 GPUs without running out of memory.
 - define `--gradient_aggregation_steps`. This will split the minibatch that is sent to a device into X steps, and the gradients from all steps are accumulated. For example, defining `--gradient_aggregation_steps 4` on a training run with 1 device should result in the same memory consumption as `--gradient_aggregation_steps 1` with 4 devices.
