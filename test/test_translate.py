#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        for i, (line, line2) in enumerate(zip(open(output1).readlines(), open(output2).readlines())):
            self.assertEqual(line.strip(), line2.strip())

    def get_settings(self):
        """
        Initialize and customize settings.
        """
        translation_settings = TranslationSettings()
        translation_settings.input = open('../../en-de/in')
        translation_settings.output = open('../../en-de/out','w')
        translation_settings.models = ["model.npz"]
        translation_settings.beam_size = 12
        translation_settings.normalization_alpha = 1.0
        return translation_settings

    # English-German WMT16 system, no dropout
    def test_ende(self):
        os.chdir('models/en-de/')
        translate(settings=self.get_settings())
        os.chdir('../..')
        self.outputEqual('en-de/ref2','en-de/out')


if __name__ == '__main__':
    unittest.main()
