#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from chrf import CharacterFScorer

class TestCharacterFScoreReference(unittest.TestCase):
    """
    Regression tests for SmoothedBleuReference
    """
    @staticmethod
    def tokenize(sentence):
        return sentence.split(" ")
    def test_identical_segments(self):
        segment = self.tokenize("Consistency is the last refuge of the unimaginative")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment)
        self.assertEqual(scorer.score(segment), 1.0)
    def test_completely_different_segments(self):
        segment_a = self.tokenize("A A A")
        segment_b = self.tokenize("B B B")
        scorer = CharacterFScorer('n=3,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)
    def test_almost_correct(self):
        segment_a = self.tokenize("foo bar")
        segment_b = self.tokenize("foo bat")
        scorer = CharacterFScorer('n=6,beta=1')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.5)
    def test_almost_correct_asym(self):
        segment_a = self.tokenize("foofoofoot")
        segment_b = self.tokenize("foofoofoo")     
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 10*((1*0.8)/((9*1)+0.8)))
    def test_too_short_correct(self):
        segment_a = self.tokenize("foofo")
        segment_b = self.tokenize("foofo")     
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 1.0)
    def test_too_short_incorrect_0(self):
        segment_a = self.tokenize("foofo")
        segment_b = self.tokenize("foobar")     
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
    def test_too_short_incorrect_1(self):
        segment_a = self.tokenize("foobar")
        segment_b = self.tokenize("foofo")     
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)    

if __name__ == '__main__':
    unittest.main()
