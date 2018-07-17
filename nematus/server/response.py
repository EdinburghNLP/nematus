#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines the abstract response format for Nematus server.
"""

from abc import ABCMeta, abstractmethod

class TranslationResponse(object):
    """
    Abstract translation response base class.
    """
    __metaclass__ = ABCMeta

    STATUS_OK = 0
    STATUS_ERROR = 1

    def __init__(self, status, segments, word_alignments=None, word_probabilities=None):
        """
        Initialises a translation response.

        @type segments: list(str)
        @param segments: the translated segments to be included.
        """
        self._content_type = "application/json"
        self._status = status
        self._segments = segments
        self._word_alignments = word_alignments
        self._word_probabilities = word_probabilities
        self._response = self._format()

    @abstractmethod
    def _format(self):
        """
        Formats this translation response.
        """
        pass # to be implemented in subclasses

    def __repr__(self):
        """
        Returns the raw body of this translation response.
        """
        return self._format()

    def get_content_type(self):
        return self._content_type
