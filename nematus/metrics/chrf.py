#!/usr/bin/env python

from metrics.scorer import Scorer
from metrics.reference import Reference

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
        if not 'n' in list(self._arguments.keys()):
            self._arguments['n'] = 6
        # use beta = 1 by default (recommendation by Maja Popovic for generative modelling)
        if not 'beta' in list(self._arguments.keys()):
            self._arguments['beta'] = 1

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

    def __init__(self, reference_tokens, n=6, beta=1):
        """
        @param reference the reference translation that hypotheses shall be
                         scored against.
        @param n         maximum character n-gram order to consider.
        @param beta      algorithm paramater beta (interpolation weight, needs to be > 0).
        """
        if beta <= 0:
            raise ValueError("Value of beta needs to be larger than zero!")
        
        Reference.__init__(self, reference_tokens)
        self.n = n
        self.max_order = n
        self.beta_squared = beta ** 2
        
        # The paper specifies that whitespace is ignored, but for a training objective,
        #it's perhaps better to leave it in. According to the paper, it makes no
        #difference in practise for scoring.
        self._reference_string = " ".join(reference_tokens).strip()
                
        # Get n-grams from reference:
        self._reference_ngrams = self._get_ngrams(self._reference_string, self.n)
        
    def _get_ngrams(self, tokens, n):
        """
        Extracts all n-grams up to order @param n from a list of @param tokens.
        """     
        n_grams_dict = {}
        length = len(tokens)
        #If the reference is shorter than n characters, insist on an exact match:
        if len(tokens) < n:
            self.max_order = len(tokens)
        m = 1
        while m <= n: #n-gram order
            i = m
            n_grams_list = []
            order_dict = {}
            while (i <= length):
                n_grams_list.append(tokens[i-m:i])
                i += 1            
            for ngr in n_grams_list:
                order_dict[ngr] = order_dict.setdefault(ngr,0) + 1
            n_grams_dict[m] = order_dict
            m += 1
        return n_grams_dict

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.

        @return the sentence-level ChrF score: 1.0 is best, 0.0 worst.
        """
        #See comment above on treating whitespace.
        hypothesis_string = " ".join(hypothesis_tokens).strip()
        
        #If the hypothesis or the reference is empty, insist on an exact match:
        if len(self._reference_string) < 1 or len(hypothesis_string) < 1:
            if hypothesis_string == self._reference_string:
                return 1.0
            else:
                return 0.0
        
        hypothesis_ngrams = self._get_ngrams(hypothesis_string, self.n)
        
        #Calculate character precision:
        chrP = 0.0
        chrR = 0.0
        for m in range(1,self.n+1):
            hyp_count = 0.0
            count_total = 0.0
            count_in = 0.0
            for ngr in hypothesis_ngrams[m]:
                hyp_count = hypothesis_ngrams[m][ngr]
                count_total += hyp_count
                if ngr in self._reference_ngrams[m]:
                    count_in += min(hyp_count, self._reference_ngrams[m][ngr])
            #Catch division by zero:
            if count_total == 0.0:
                chrP += 0.0
            else:
                chrP += count_in / count_total    
        #average chrP over n-gram orders:        
        chrP = chrP / float(self.max_order)
        
        #Calculate character recall:
        for m in range(1,self.n+1):
            ref_count = 0.0
            count_total = 0.0
            count_in = 0.0
            for ngr in self._reference_ngrams[m]:
                ref_count = self._reference_ngrams[m][ngr]
                count_total += ref_count
                if ngr in hypothesis_ngrams[m]:
                    count_in += min(ref_count, hypothesis_ngrams[m][ngr])
            #Catch division by zero:
            if count_total == 0.0:
                chrR += 0.0
            else:    
                chrR += count_in/count_total
        #average chrR over n-gram orders:
        chrR = chrR / float(self.max_order)
                
        #Catch division by zero:
        if chrP == 0.0 and chrR == 0.0:
            return 0.0
        return (1 + self.beta_squared) * (chrP*chrR) / ((self.beta_squared * chrP) + chrR)
