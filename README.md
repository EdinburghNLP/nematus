NEMATUS
-------
This fork migrates Nematus-Tensorflow to Python 3 and Windows
(This branch uses a Tensorflow backend. See https://github.com/EdinburghNLP/nematus/tree/master for the Theano-based version of Nematus,
which currently supports more architecture variants.)

Attention-based encoder-decoder model for neural machine translation

This package is based on the dl4mt-tutorial by Kyunghyun Cho et al. ( https://github.com/nyu-dl/dl4mt-tutorial ).
It was used to produce top-scoring systems at the WMT 16 shared translation task.

The changes to Nematus include:

  - new architecture variants for better performance:
     - tied embeddings (Press and Wolf, 2016) https://arxiv.org/abs/1608.05859
     - layer normalisation (Ba et al, 2016) https://arxiv.org/abs/1607.06450

 - improvements to scoring and decoding:
     - n-best output for decoder
     - scripts for scoring (given parallel corpus) and rescoring (of n-best output)

 - usability improvements:
     - command line interface for training
     - vocabulary files and model parameters are stored in JSON format (backward-compatible loading)
     - server mode

see changelog for more info.


SUPPORT
-------

For general support requests, there is a Google Groups mailing list at https://groups.google.com/d/forum/nematus-support . You can also send an e-mail to nematus-support@googlegroups.com .


INSTALLATION
------------

Nematus requires the following packages:

 - Python >= 2.7
 - tensorflow

To install tensorflow, we recommend following the steps at:
  ( https://www.tensorflow.org/install/ )

the following packages are optional, but *highly* recommended

 - CUDA >= 7  (only GPU training is sufficiently fast)
 - cuDNN >= 4 (speeds up training substantially)


DOCKER USAGE
------------

You can also create docker image by running following command, where you change `suffix` to either `cpu` or `gpu`:

`docker build -t nematus-docker -f Dockerfile.suffix .`

To run a CPU docker instance with the current working directory shared with the Docker container, execute:

``docker run -v `pwd`:/playground -it nematus-docker``

For GPU you need to have nvidia-docker installed and run:

``nvidia-docker run -v `pwd`:/playground -it nematus-docker``


TRAINING SPEED
--------------

Training speed depends heavily on having appropriate hardware (ideally a recent NVIDIA GPU),
and having installed the appropriate software packages.

To test your setup, we provide some speed benchmarks with `test/test_train.sh',
on an Intel Xeon CPU E5-2620 v4, with a Nvidia GeForce GTX Titan X (Pascal) and CUDA 9.0:


GPU, CuDNN 5.1, tensorflow 1.0.1:

  CUDA_VISIBLE_DEVICES=0 ./test_train.sh

>> 225.25 sentenses/s

 
USAGE INSTRUCTIONS
------------------

All of the scripts below can be run with `--help` flag to get usage information.

#### `nematus/nmt.py` : use to train a new model

#### `nematus/translate.py` : use an existing model to translate a source text

#### `nematus/score.py` : use an existing model to score a parallel corpus

#### `nematus/rescore.py` : use an existing model to rescore an n-best list.

The n-best list is assumed to have the same format as Moses:

    sentence-ID (starting from 0) ||| translation ||| scores

new scores will be appended to the end. 

#### `nematus/theano_tf_convert.py` : convert an existing theano model to a tensorflow model

If you have a Theano model (model.npz) with network architecture features that are currently
supported then you can convert it into a tensorflow model using `nematus/theano_tf_convert.py`.


PUBLICATIONS
------------

if you use Nematus, please cite the following paper:

Rico Sennrich, Orhan Firat, Kyunghyun Cho, Alexandra Birch, Barry Haddow, Julian Hitschler, Marcin Junczys-Dowmunt, Samuel Läubli, Antonio Valerio Miceli Barone, Jozef Mokry and Maria Nadejde (2017): Nematus: a Toolkit for Neural Machine Translation. In Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics, Valencia, Spain, pp. 65-68.

```
@InProceedings{sennrich-EtAl:2017:EACLDemo,
  author    = {Sennrich, Rico  and  Firat, Orhan  and  Cho, Kyunghyun  and  Birch, Alexandra  and  Haddow, Barry  and  Hitschler, Julian  and  Junczys-Dowmunt, Marcin  and  L\"{a}ubli, Samuel  and  Miceli Barone, Antonio Valerio  and  Mokry, Jozef  and  Nadejde, Maria},
  title     = {Nematus: a Toolkit for Neural Machine Translation},
  booktitle = {Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {65--68},
  url       = {http://aclweb.org/anthology/E17-3017}
}
```

the code is based on the following model:

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015): Neural Machine Translation by Jointly Learning to Align and Translate, Proceedings of the International Conference on Learning Representations (ICLR).

please refer to the Nematus paper for a description of implementation differences


ACKNOWLEDGMENTS
---------------
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreements 645452 (QT21), 644333 (TraMOOC), 644402 (HimL) and 688139 (SUMMA).