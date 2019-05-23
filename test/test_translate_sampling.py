#!/usr/bin/env python3

import sys
import os
import unittest
import logging

sys.path.append(os.path.abspath('../nematus'))
from translate import main as translate
from settings import TranslationSettings
from test_utils import load_wmt16_model

level = logging.DEBUG
logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

class TestTranslate(unittest.TestCase):
    """
    Regression tests for translation with WMT16 models
    """

    def setUp(self):
        """
        Download pre-trained models
        """
        load_wmt16_model('en','de')

    def outputEqual(self, output1, output2):
        """given two translation outputs, check that output string is identical
        """
        with open(output1, 'r', encoding='utf-8') as out1, \
             open(output2, 'r', encoding='utf-8') as out2:
            for (line1, line2) in zip(out1.readlines(), out2.readlines()):
                self.assertEqual(line1.strip(), line2.strip())

    # English-German WMT16 system, no dropout
    def test_ende(self):
        with open('en-de/in', 'r', encoding='utf-8') as in_file, \
             open('en-de/out', 'w', encoding='utf-8') as out_file:
            os.chdir('models/en-de/')
            settings = TranslationSettings()
            settings.input = in_file
            settings.output = out_file
            settings.models = ["model.npz"]
            settings.beam_size = 12
            settings.normalization_alpha = 1.0
            settings.translation_strategy = 'sampling'
            settings.sampling_temperature = 0.4
            translate(settings=settings)
            os.chdir('../..')
        self.outputEqual('en-de/ref2','en-de/out')


if __name__ == '__main__':
    unittest.main()
