#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    @staticmethod
    def get_settings():
        scorer_settings = ScorerSettings()
        scorer_settings.models = ['model.npz']
        scorer_settings.b = 80
        scorer_settings.normalization_alpha = 1.0
        return scorer_settings

    def scoreEqual(self, output1, output2):
        """given two files with translation scores, check that probabilities are equal within rounding error.
        """
        for i, (line, line2) in enumerate(zip(open(output1).readlines(), open(output2).readlines())):
            self.assertAlmostEqual(float(line.split()[-1]), float(line2.split()[-1]), 5)

    # English-German WMT16 system, no dropout
    def test_ende(self):
        scorer_settings = self.get_settings()
        os.chdir('models/en-de/')
        score(open('../../en-de/in'), open('../../en-de/references'), open('../../en-de/out_score','w'), scorer_settings)
        os.chdir('../..')
        self.scoreEqual('en-de/ref_score', 'en-de/out_score')

if __name__ == '__main__':
    unittest.main()
