#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class Reference:
    """
    Abstract base class for re-usable translation reference. Hypotheses can be
    scored against this reference through the evaluation metric implemented in
    its `score` function.
    """

    __metaclass__ = ABCMeta #abstract base class

    def __init__(self, reference_tokens):
        """
        @param reference the reference translation that hypotheses shall be
                         scored against.
        """
        self._reference_tokens = reference_tokens
        #additional (metric-specific) parameters to be defined in subclass

    @abstractmethod
    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.
        """
        pass #to be implemented in sublcass

    def score_matrix(self, hypothesis_matrix):
        """
        Scores every hypothesis in @param hypotheses against this reference.
        @param hypothesis_matrix an iterable of iterables of tokens.
        """
        return [self.score(hypothesis_tokens) for hypothesis_tokens in hypothesis_matrix]
