#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sentence_bleu import SentenceBleuScorer
from meteor import MeteorScorer

class ScorerProvider:
    """
    Parses a config string and returns a matching scorer object with the given
    parameters.
    """
    #from bleu import SentenceBleuScorer

    def __init__(self):
        pass

    def get(self, config_string):
        """
        Returns a scorer matching the metric and parameters defined in @param
        config string.

        Example: ScorerProvider.get("BLEU n=4") returns a SmoothedBleuScorer
                 object that considers n-gram precision up to n=4.
        """
        scorer, arguments = config_string.split(" ", 1)
        if scorer == 'SENTENCEBLEU':
            return SentenceBleuScorer(arguments)
        elif scorer == 'METEOR':
            return MeteorScorer(arguments)
        elif scorer == 'BEER':
            return MeteorScorer(arguments)
        # add other scorers here
        else:
            raise NotImplementedError("No such scorer: %s" % scorer)
