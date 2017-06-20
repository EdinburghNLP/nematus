#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs Nematus as a Web Server.
"""

import json
import pkg_resources
import sys

from bottle import Bottle, request, response

from server.response import TranslationResponse
from server.api.provider import request_provider, response_provider
from console import ConsoleInterfaceServer
from translate import Translator

class NematusServer(object):
    """
    Keeps a Nematus model in memory to answer http translation requests.
    """

    STATUS_LOADING = 'loading'
    STATUS_OK = 'ok'

    def __init__(self, server_settings, decoder_settings):
        """
        Loads a translation model and initialises the webserver.

        @param startup args: as defined in `console.py`
        """
        self._style = server_settings.style
        self._host = server_settings.host
        self._port = server_settings.port
        self._debug = decoder_settings.verbose
        self._models = decoder_settings.models
        self._num_processes = decoder_settings.num_processes
        self._device_list = decoder_settings.device_list
        self._status = self.STATUS_LOADING
        # start webserver
        self._server = Bottle()
        # start translation workers
        self._translator = Translator(decoder_settings)
        self._status = self.STATUS_OK

    def status(self):
        """
        Reports on the status of this translation server.
        """
        response_data = {
            'status': self._status,
            'models': self._models,
            'version': pkg_resources.require("nematus")[0].version,
            'service': 'nematus',
        }
        response.content_type = "application/json"
        return json.dumps(response_data)

    def translate(self):
        """
        Processes a translation request.
        """
        translation_request = request_provider(self._style, request)
        sys.stderr.write("REQUEST - " + repr(translation_request) + "\n")

        translations = self._translator.translate(
            translation_request.segments,
            translation_request.settings
        )
        response_data = {
            'status': TranslationResponse.STATUS_OK,
            'segments': [translation.target_words for translation in translations],
            'word_alignments': [translation.get_alignment_json(as_string=False) for translation in translations] if translation_request.settings.get_alignment else None,
            'word_probabilities': [translation.target_probs for translation in translations] if translation_request.settings.get_word_probs else None,
        }
        translation_response = response_provider(self._style, **response_data)
        sys.stderr.write("RESPONSE - " + repr(translation_response) + "\n")

        response.content_type = translation_response.get_content_type()
        return repr(translation_response)

    def start(self):
        """
        Starts the webserver.
        """
        self._route()
        self._server.run(host=self._host, port=self._port, debug=self._debug, server='paste')

    def _route(self):
        """
        Routes webserver paths to functions.
        """
        self._server.route('/status', method="GET", callback=self.status)
        self._server.route('/translate', method="POST", callback=self.translate)


if __name__ == "__main__":
    parser = ConsoleInterfaceServer()
    server_settings = parser.get_server_settings()
    decoder_settings = parser.get_decoder_settings()

    server = NematusServer(server_settings, decoder_settings)
    server.start()
