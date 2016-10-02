#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from math import exp
from operator import mul
from collections import defaultdict

from scorer import Scorer
from reference import Reference

class SentenceBleuScorer(Scorer):
    """
    Scores SmoothedBleuReference objects.
    """

    def __init__(self, argument_string):
        """
        Initialises metric-specific parameters.
        """
        Scorer.__init__(self, argument_string)
        # use n-gram order of 4 by default
        if not 'n' in self._arguments.keys():
            self._arguments['n'] = 4

    def set_reference(self, reference_tokens):
        """
        Sets the reference against hypotheses are scored.
        """
        self._reference = SentenceBleuReference(
            reference_tokens,
            self._arguments['n']
        )

class SentenceBleuReference(Reference):
    """
    Smoothed sentence-level BLEU as as proposed by Lin and Och (2004).
    Implemented as described in (Chen and Cherry, 2014).
    """

    def __init__(self, reference_tokens, n=4):
        """
        @param reference the reference translation that hypotheses shall be
                         scored against. Must be an iterable of tokens (any
                         type).
        @param n         maximum n-gram order to consider.
        """
        Reference.__init__(self, reference_tokens)
        self.n = n
        # preprocess reference
        self._reference_length = len(self._reference_tokens)
        self._reference_ngrams = self._get_ngrams(self._reference_tokens, self.n)

    def _get_ngrams(self, tokens, max_n):
        """
        Extracts all n-grams of order 1 up to (and including) @param max_n from
        a list of @param tokens.
        """
        n_grams = []
        for n in range(1, max_n+1):
            n_grams.append(defaultdict(int))
            for n_gram in zip(*[tokens[i:] for i in range(n)]):
                n_grams[n-1][n_gram] += 1
        return n_grams

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.

        @return the smoothed sentence-level BLEU score: 1.0 is best, 0.0 worst.
        """
        def product(iterable):
            return reduce(mul, iterable, 1)
        def ngram_precisions(ref_ngrams, hyp_ngrams):
            precisions = []
            for n in range(1, self.n+1):
                overlap = 0
                for ref_ngram, ref_ngram_count in ref_ngrams[n-1].iteritems():
                    if ref_ngram in hyp_ngrams[n-1]:
                        overlap += min(ref_ngram_count, hyp_ngrams[n-1][ref_ngram])
                hyp_length = max(0, len(hypothesis_tokens)-n+1)
                if n >= 2:
                    # smoothing as proposed by Lin and Och (2004),
                    # implemented as described in (Chen and Cherry, 2014)
                    overlap += 1
                    hyp_length += 1
                precisions.append(overlap/hyp_length if hyp_length > 0 else 0.0)
            return precisions
        def brevity_penalty(ref_length, hyp_length):
            return min(1.0, exp(1-(ref_length/hyp_length if hyp_length > 0 else 0.0)))
        # preprocess hypothesis
        hypothesis_length = len(hypothesis_tokens)
        hypothesis_ngrams = self._get_ngrams(hypothesis_tokens, self.n)
        # calculate n-gram precision for all orders
        np = ngram_precisions(self._reference_ngrams, hypothesis_ngrams)
        # calculate brevity penalty
        bp = brevity_penalty(self._reference_length, hypothesis_length)
        # compose final BLEU score
        return product(np)**(1/self.n) * bp
