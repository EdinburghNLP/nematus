#!/usr/bin/env python3

import sys
import os
import unittest
import logging

sys.path.append(os.path.abspath('../nematus'))
from score import main as score
from settings import ScorerSettings
from test_utils import load_wmt16_model

level = logging.DEBUG
logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

class TestScore(unittest.TestCase):
    """
    Regression tests for scoring with WMT16 models
    """

    def setUp(self):
        """
        Download pre-trained models
        """
        load_wmt16_model('en','de')

    def scoreEqual(self, output1, output2):
        """Given two files with translation scores, check that probabilities
           are equal within rounding error.
        """
        with open(output1, 'r', encoding='utf-8') as out1, \
             open(output2, 'r', encoding='utf-8') as out2:
            for (line1, line2) in zip(out1.readlines(), out2.readlines()):
                score1 = float(line1.split()[-1])
                score2 = float(line2.split()[-1])
                self.assertAlmostEqual(score1, score2, places=5)

    # English-German WMT16 system, no dropout
    def test_ende(self):
        os.chdir('models/en-de/')
        with open('../../en-de/in', 'r', encoding='utf-8') as in_file, \
             open('../../en-de/references', 'r', encoding='utf-8') as ref_file, \
             open('../../en-de/out_score', 'w', encoding='utf-8') as score_file:
            settings = ScorerSettings()
            settings.models = ['model.npz']
            settings.minibatch_size = 80
            settings.normalization_alpha = 1.0
            score(in_file, ref_file, score_file, settings)
        os.chdir('../..')
        self.scoreEqual('en-de/ref_score', 'en-de/out_score')


if __name__ == '__main__':
    unittest.main()
