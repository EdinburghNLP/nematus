#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class Scorer:
    """
    Abstract base class for MT evaluation metric. Can be passed on to a
    Reference for scoring translation hypotheses.
    """

    __metaclass__ = ABCMeta #abstract base class

    def __init__(self, argument_string):
        """
        @param argument_string the metric-specific parameters (such as n-gram
        order for BLEU, language for METEOR, etc.)
        """
        # parse arguments
        self._reference = None # to be set via `self.set_reference()`
        self._arguments = {}
        if argument_string:
            argument_strings = argument_string.split(",")
            for a in argument_strings:
                argument, value = a.split("=")
                argument = argument.strip()
                value = value.strip()
                try:
                    value = int(value) # change type to int if applicable
                except ValueError:
                    value = value
                self._arguments[argument] = value

    @abstractmethod
    def set_reference(self, reference_tokens):
        """
        Sets the reference against which one or many hypotheses can be scored
        via `self.score()` and `self.score_matrix()`.
        """
        pass # instantiate a Reference object and store it at self._reference

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.
        """
        return self._reference.score(hypothesis_tokens)

    def score_matrix(self, hypothesis_matrix):
        """
        Scores every hypothesis in @param hypotheses against this reference.
        @param hypothesis_matrix an iterable of iterables of tokens.
        """
        return self._reference.score_matrix(hypothesis_matrix)
