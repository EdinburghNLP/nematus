This directory contains small scripts for data processing and evaluation.
Other useful scripts and sample data is provided at https://github.com/rsennrich/wmt16-scripts


Evaluation
----------

This directory contains two evaluation scripts:

 - multi-bleu.perl (from Moses decoder) computes tokenized, case-sensitive BLEU 
   scores. This script is widely used in NMT research, but we discourage its use 
   for publication because different groups use different tokenization, which 
   biases comparisons to previous work.

   usage:
   ./multi-bleu.perl ref_file < test_file

 - multi-bleu-detok.perl expects that the reference file and output file are not 
   tokenized (untokenized reference; detokenized output). It performs tokenization 
   internally, using the tokenization routine from the NIST BLEU scorer 
   (mteval-v13a.pl). This script can be used as a plaintext alternative of 
   mteval-v13a.pl, giving the same results.

   usage:
   ./multi-bleu-detok.perl ref_file < test_file