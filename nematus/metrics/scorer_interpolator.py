#!/usr/bin/env python

from metrics.scorer import Scorer
from metrics import scorer_provider as sp

class ScorerInterpolator(Scorer):
    """
    Creates a scorer that interpolates scores from 1..n sub-scorers, e.g.,
    0.5 * SENTENCEBLEU + 0.5 * METEOR.
    """

    def __init__(self, config_string):
        """
        @param config_string example:
        `INTERPOLATE w=0.5,0.5; SENTENCEBLEU n=4; METEOR meteor_language=fr, meteor_path=/foo/bar/meteor`
        """
        self._scorers = []
        self._weights = []
        # parse arguments
        scorers = config_string.split(";")
        scorers = [scorer.strip() for scorer in scorers]
        try:
            instruction, weights = scorers[0].split("w=")
            assert instruction.strip() == "INTERPOLATE"
            weights = [float(w) for w in weights.split(',')]
            scorers = [sp.ScorerProvider().get(s) for s in scorers[1:]]
        except:
            raise SyntaxError("Ill-formated interpolation of metrics. Example of valid definition: `INTERPOLATE w=0.5,0.5`.")
        # assertions
        assert len(weights) == len(scorers)
        assert sum(weights) == 1.0
        # init scorers
        for i, scorer in enumerate(scorers):
            self._scorers.append(scorer)
            self._weights.append(weights[i])

    def set_reference(self, reference_tokens):
        """
        Sets the reference against which one or many hypotheses can be scored
        via `self.score()` and `self.score_matrix()`.
        """
        for scorer in self._scorers:
            scorer.set_reference(reference_tokens)

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis with all scorers added via `self.add_scorer`
        and interpolates the scores with the respective weights.
        """
        return sum([s.score(hypothesis_tokens) * w for w, s in zip(self._weights, self._scorers)])

    def score_matrix(self, hypothesis_matrix):
        """
        Scores every hypothesis in @param hypotheses with all scorers added via
        `self.add_scorer` and interpolates the scores with the respective
        weights.
        """
        return sum([s.score_matrix(hypothesis_matrix) * w for w, s in zip(self._weights, self._scorers)])
