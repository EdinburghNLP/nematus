#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines the abstract request format for Nematus server.
"""

from abc import ABCMeta, abstractmethod

class TranslationRequest(object):
    """
    Abstract translation request base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, request):
        """
        Initialises a translation request.

        @type raw_body: str
        @param raw_body: the POST request submitted to Nematus server.
        """
        self._request = request
        self.segments = []
        self.beam_width = 5
        self.normalize = True
        self.character_level = False
        self.n_best = 1
        self.suppress_unk = False
        self.return_word_alignment = False
        self.return_word_probabilities = False
        self._parse()

    @abstractmethod
    def _parse(self):
        """
        Parses the request's raw body. Sets or overrides
        * self.segments
        * self.beam_width
        * self.normalize
        * self.character_level
        * self.n_best
        * self.suppress_unk
        * self.return_word_alignment
        * self.return_word_probabilities
        """
        pass # to be implemented in subclasses
