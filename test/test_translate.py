#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import unittest
import requests

sys.path.append(os.path.abspath('../nematus'))
from translate import main as translate


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

    # English-German WMT16 system, no dropout
    def test_ende(self):
        os.chdir('models/en-de/')
        translate(['model.npz'], open('../../en-de/in'), open('../../en-de/out','w'), k=12, normalize=True, n_process=1, suppress_unk=True, print_word_probabilities=True)
        os.chdir('../..')
        for i, (line, line2) in enumerate(zip(open('en-de/ref').readlines(), open('en-de/out').readlines())):
        if not i % 2:
            self.assertEqual(line, line2)
        else:
            probs = map(float, line.split())
            probs2 = map(float, line.split())
            for p, p2 in zip(probs, probs2):
                self.assertAlmostEqual(p, p2, 5)

    # English-Romanian WMT16 system, dropout
#    def test_enro(self):
#        os.chdir('models/en-ro/')
#        translate(['model.npz'], open('../../en-ro/in'), open('../../en-ro/out','w'), k=12, normalize=True, n_process=1, suppress_unk=True, print_word_probabilities=True)
#        os.chdir('../..')
#        self.assertEqual(open('en-ro/ref').read(), open('en-ro/out').read())



if __name__ == '__main__':
    unittest.main()
