#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines the Nematus API for translation requests and responses.
"""

import json
from ..request import TranslationRequest
from ..response import TranslationResponse

class TranslationRequestNematus(TranslationRequest):
    def _parse(self):
        # never produce search graph
        self.get_search_graph = False

        request = self._request.json
        if 'segments' in request:
            self.segments = [' '.join(tokens) for tokens in request['segments']]
        if 'beam_width' in request:
            self.settings.beam_width = request['beam_width']
        if 'normalize' in request:
            self.settings.normalization_alpha = request['normalize']
        if 'character_level' in request:
            self.settings.char_level = request['character_level']
        if 'suppress_unk' in request:
            self.settings.suppress_unk = request['suppress_unk']
        if 'return_word_alignment' in request:
            self.settings.get_alignment = request['return_word_alignment']
        if 'return_word_probabilities' in request:
            self.settings.get_word_probs = request['return_word_probabilities']

    def _format(self):
        request = {
            'id': str(self.settings.request_id),
            'data': [segment for segment in self.segments]
        }
        return json.dumps(request)

class TranslationResponseNematus(TranslationResponse):
    def _format(self):
        response = {
            'status': '',
            'data': [],
        }
        if self._status == self.STATUS_OK:
            response['status'] = 'ok'
            for i, translation in enumerate(self._segments):
                segment = {'translation': translation}
                if self._word_alignments:
                    segment['word_alignment'] = self._word_alignments[i]
                if self._word_probabilities:
                    segment['word_probabilities'] = self._word_probabilities[i]
                response['data'].append(segment)
        else:
            response['status'] = 'error'
        return json.dumps(response)
