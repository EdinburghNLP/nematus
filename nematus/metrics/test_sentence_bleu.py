#!/usr/bin/env python

import unittest

from metrics.sentence_bleu import SentenceBleuScorer

class TestSentenceBleuReference(unittest.TestCase):
    """
    Regression tests for SmoothedBleuReference
    """
    @staticmethod
    def tokenize(sentence):
        return sentence.split(" ")
    def test_identical_segments(self):
        segment = self.tokenize("Consistency is the last refuge of the unimaginative")
        scorer = SentenceBleuScorer('n=4')
        scorer.set_reference(segment)
        self.assertEqual(scorer.score(segment), 1.0)
    def test_completely_different_segments(self):
        segment_a = self.tokenize("A A A")
        segment_b = self.tokenize("B B B")
        scorer = SentenceBleuScorer('n=4')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)
    def test_clipping(self):
        segment_a = self.tokenize("The very nice man")
        segment_b = self.tokenize("man man man man")
        scorer = SentenceBleuScorer('n=1')
        scorer.set_reference(segment_a)
        self.assertNotEqual(scorer.score(segment_b), 1.0)

if __name__ == '__main__':
    unittest.main()
