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
        segment_a = self.tokenize("AAAAAA")
        segment_b = self.tokenize("BBBB")
        scorer = CharacterFScorer('n=3,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)
    def test_empty_string(self):
        segment_a = self.tokenize("")
        segment_b = self.tokenize("")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 1.0)
    def test_one_character_empty_string(self):
        segment_a = self.tokenize("A")
        segment_b = self.tokenize("")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)
    def test_empty_string_one_character(self):
        segment_a = self.tokenize("")
        segment_b = self.tokenize("A")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.0)
    def test_half_right(self):
        segment_a = self.tokenize("AB")
        segment_b = self.tokenize("AA")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 0.25)                     
    def test_one_character(self):
        segment_a = self.tokenize("A")
        segment_b = self.tokenize("A")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual(scorer.score(segment_b), 1.0)
    def test_almost_correct(self):
        segment_a = self.tokenize("risk assessment has to be undertaken by those who are qualified and expert in that area - that is the scientists .")
        segment_b = self.tokenize(" risk assessment must be made of those who are qualified and expertise in the sector - these are the scientists .")
        scorer = CharacterFScorer('n=6,beta=3')
        scorer.set_reference(segment_a)
        self.assertEqual('{0:.12f}'.format(scorer.score(segment_b)), "0.652414427449")
    
if __name__ == '__main__':
    unittest.main()
