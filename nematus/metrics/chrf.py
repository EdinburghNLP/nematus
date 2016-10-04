#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scorer import Scorer
from reference import Reference

class CharacterFScorer(Scorer):
    """
    Scores CharacterFScoreReference objects.
    """

    def __init__(self, argument_string):
        """
        Initialises metric-specific parameters.
        """
        Scorer.__init__(self, argument_string)
        # use character n-gram order of 4 by default
        if not 'n' in self._arguments.keys():
            self._arguments['n'] = 6
        # use beta = 3 by default
        if not 'beta' in self._arguments.keys():
            self._arguments['beta'] = 3

    def set_reference(self, reference_tokens):
        """
        Sets the reference against hypotheses are scored.
        """
        self._reference = CharacterFScoreReference(
            reference_tokens,
            self._arguments['n'],
            self._arguments['beta']
        )

class CharacterFScoreReference(Reference):
    """
    References for Character F-Score, as proposed by Popovic (2015): http://www.statmt.org/wmt15/pdf/WMT49.pdf
    """

    def __init__(self, reference_tokens, n=6, beta=3):
        """
        @param reference the reference translation that hypotheses shall be
                         scored against.
        @param n         character n-gram order to consider.
        @param beta      algorithm paramater beta (interpolation weight, needs to be > 0).
        """
        if beta <= 0:
            raise ValueError("Value of beta needs to be larger than zero!")
        
        Reference.__init__(self, reference_tokens)
        self.n = n
        self.beta_squared = beta ** 2
        
        # The paper specifies that whitespace is ignored, but for a training objective,
        #it's perhaps better to leave it in. According to the paper, it makes no
        #difference in practise for scoring.
        self._reference_string = " ".join(reference_tokens)
        
        # Get n-grams from reference:
        self._reference_ngrams, self._reference_ngrams_unique = self._get_ngrams(self._reference_string, self.n)
        
        #Set number of reference n-grams:
        self._reference_n_gram_counter = len(self._reference_ngrams)
        
        #Initialize flag for comparisons where either reference or hypothesis is shorter than n:
        self._too_short = False

    def _get_ngrams(self, tokens, n):
        """
        Extracts all n-grams of order @param n from a list of @param tokens.
        """
        n_grams = []
        length = len(tokens)
        #If the reference is shorter than n characters, insist on an exact match:
        if len(tokens) < n:
            self._too_short = True
            return n_grams, set(n_grams)
        i = n
        while (i <= length):
            n_grams.append(tokens[i-n:i])
            i += 1
        return n_grams, set(n_grams)

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.

        @return the sentence-level ChrF score: 1.0 is best, 0.0 worst.
        """
        
        #See comment above on treating whitespace.
        hypothesis_string = " ".join(hypothesis_tokens)
        
        #If the hypothesis is shorter than n characters, insist on an exact match:
        if len(hypothesis_string) < self.n:
            self._too_short = True
        if self._too_short:
            if hypothesis_string == self._reference_string:
                return 1.0
            else:
                return 0.0
 
        hypothesis_ngrams, hypothesis_ngrams_unique = self._get_ngrams(hypothesis_string, self.n)
        
        
        #Calculate character precision:
        count_in = 0.0
        count_total = len(hypothesis_ngrams)
        #Check if we have any n-grams at all:
        if count_total == 0:
            return 0
        for ngr in hypothesis_ngrams:
            if ngr in self._reference_ngrams_unique:
                count_in += 1.0
        chrP = count_in / count_total
        
        #Calculate character recall:
        count_in = 0.0
        for ngr in self._reference_ngrams:
            if ngr in hypothesis_ngrams_unique:
                count_in += 1.0
        chrR = count_in/self._reference_n_gram_counter
        #Catch division by zero:
        if chrP == 0 and chrR == 0:
            return 0
        
        return (1 + self.beta_squared) * (chrP * chrR /((self.beta_squared * chrP) + chrR))