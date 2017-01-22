# -*- coding: utf8 -*-
'''
This script is to replace the unknown words in target sentences with their aligned words in source sentences.
Args: 
	- input: an alignment file produced by translating with the option '--output_alignment'
	- output: output text file
	- unknown word token (optional): a string, default="UNK"
To use:
	python copy_unknown_words.py -i translation.txt -o updated_translation.txt -u 'UNK'
'''

import json
import numpy
import argparse
import sys

''' 
Example input file:
0 ||| das ist ein Test . ||| 0 ||| this is a UNK . ||| 6 6
0.723781 0.0561881 0.0652739 0.0888658 0.0159646 0.0499262
0.0250772 0.728351 0.105699 0.0764411 0.0245384 0.0398933
0.0257915 0.0667947 0.543118 0.177978 0.020311 0.166007
0.000306134 0.0161435 0.025201 0.937249 0.00364889 0.0174515
0.0116866 0.195885 0.0383414 0.0331976 0.437992 0.282897
0.0121966 0.00570636 0.00524746 0.014052 0.0325562 0.930241
'''

def copy_unknown_words(filename, out_filename, unk_token):
    for line in filename:
        items = line.split(' ||| ')
        if len(items) > 1:
            src = items[1].split()
            target = items[3].split()
            alignments = []
        elif line.strip():
            alignment = map(float,line.split())
            hard_alignment = numpy.argmax(alignment, axis=0)
            alignments.append(hard_alignment)
        elif line == '\n':
            print alignments
            for i, word in enumerate(target):
                if word == unk_token:
                    target[i] = src[alignments[i]]
            out_filename.write(' '.join(target) + '\n')


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                                                metavar='PATH', default=sys.stdin,
                                                help='''Input text file (produced by decoding with \'--output_alignment\')''')
        parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                                                default=sys.stdout, metavar='PATH',
                                                help="Output file (default: standard output)")
        parser.add_argument('--unknown', '-u', type=str, nargs = '?', default="UNK",
                                                help='Unknown token to be replaced (default: "UNK")')

        args = parser.parse_args()

        copy_unknown_words(args.input, args.output, args.unknown)