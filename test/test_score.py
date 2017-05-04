#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import unittest
import requests

sys.path.append(os.path.abspath('../nematus'))
from score import main as score


def load_wmt16_model(src, target):
        path = os.path.join('models', '{0}-{1}'.format(src,target))
        try:
            os.makedirs(path)
        except OSError:
            pass
        for filename in ['model.npz', 'model.npz.json', 'vocab.{0}.json'.format(src), 'vocab.{0}.json'.format(target)]:
            if not os.path.exists(os.path.join(path, filename)):
                r = requests.get('http://data.statmt.org/rsennrich/wmt16_systems/{0}-{1}/'.format(src,target) + filename, stream=True)
                with open(os.path.join(path, filename), 'wb') as f:
                    for chunk in r.iter_content(1024**2):
                        f.write(chunk)

class TestTranslate(unittest.TestCase):
    """
    Regression tests for translation with WMT16 models
    """

    def setUp(self):
        """
        Download pre-trained models
        """
        load_wmt16_model('en','de')
        load_wmt16_model('en','ro')

    def scoreEqual(self, output1, output2):
        """given two files with translation scores, check that probabilities are equal within rounding error.
        """
        for i, (line, line2) in enumerate(zip(open(output1).readlines(), open(output2).readlines())):
            self.assertAlmostEqual(float(line.split()[-1]), float(line2.split()[-1]), 5)

    # English-German WMT16 system, no dropout
    def test_ende(self):
        os.chdir('models/en-de/')
        score(['model.npz'], open('../../en-de/in'), open('../../en-de/references'), open('../../en-de/out_score','w'), normalization_alpha)
        os.chdir('../..')
        self.scoreEqual('en-de/ref_score', 'en-de/out_score')

    # English-Romanian WMT16 system, dropout
    def test_enro(self):
        os.chdir('models/en-ro/')
        score(['model.npz'], open('../../en-ro/in'), open('../../en-ro/references'), open('../../en-ro/out_score','w'), normalization_alpha)
        os.chdir('../..')
        self.scoreEqual('en-ro/ref_score', 'en-ro/out_score')



if __name__ == '__main__':
    unittest.main()
