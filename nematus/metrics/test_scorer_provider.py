#!/usr/bin/env python

import unittest

from metrics.scorer_provider import ScorerProvider
from metrics.sentence_bleu import SentenceBleuScorer

class TestScorerProvider(unittest.TestCase):
    """
    Regression tests for ScorerProvider
    """
    @staticmethod
    def tokenize(sentence):
        return sentence.split(" ")

    def test_single_metric(self):
        config_string = "SENTENCEBLEU n=4"
        segment = self.tokenize("Consistency is the last refuge of the unimaginative")
        reference_scorer = SentenceBleuScorer('n=4')
        provided_scorer = ScorerProvider().get(config_string)
        reference_scorer.set_reference(segment)
        provided_scorer.set_reference(segment)
        self.assertEqual(
            reference_scorer.score(segment),
            provided_scorer.score(segment)
        )

    def test_interpolated_metrics(self):
        config_string = "INTERPOLATE w=0.3,0.7; SENTENCEBLEU n=4; SENTENCEBLEU n=4"
        segment = self.tokenize("Consistency is the last refuge of the unimaginative")
        reference_scorer = SentenceBleuScorer('n=4')
        provided_scorer = ScorerProvider().get(config_string) # interpolating BLEU with BLEU should obviously result in the same as just using a single BLEU scorer
        reference_scorer.set_reference(segment)
        provided_scorer.set_reference(segment)
        self.assertEqual(
            reference_scorer.score(segment),
            provided_scorer.score(segment)
        )


if __name__ == '__main__':
    unittest.main()
