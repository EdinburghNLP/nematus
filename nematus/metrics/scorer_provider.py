#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scorer_interpolator as si

from sentence_bleu import SentenceBleuScorer
from meteor import MeteorScorer
from beer import BeerScorer
from chrf import CharacterFScorer

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

        If more than one metrics are provided (separated by `;`),
        an interpolated scorer will be returned.

        Example: ScorerProvider.get("INTERPOLATE w=0.5,0.5; SENTENCEBLEU n=4; METEOR meteor_language=fr, meteor_path=/foo/bar/meteor")
                 returns an InterpolatedScorer object that scores hypotheses
                 using 0.5 * bleu_score + 0.5 * meteor_score.
        """
        # interpolation
        if config_string.startswith("INTERPOLATE"):
            return si.ScorerInterpolator(config_string)
        try:
            scorer, arguments = config_string.split(" ", 1)
        except ValueError:
            scorer = config_string
            arguments = ''
        if scorer == 'SENTENCEBLEU':
            return SentenceBleuScorer(arguments)
        elif scorer == 'METEOR':
            return MeteorScorer(arguments)
        elif scorer == 'BEER':
            return BeerScorer(arguments)
        elif scorer == 'CHRF':
            return CharacterFScorer(arguments)
        # add other scorers here
        else:
            raise NotImplementedError("No such scorer: %s" % scorer)
