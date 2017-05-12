#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs Nematus as a Web Server.
"""

import json
import pkg_resources

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

    def __init__(self, startup_args):
        """
        Loads a translation model and initialises the webserver.

        @param startup args: as defined in `console.py`
        """
        self._models = startup_args.models
        self._style = startup_args.style
        self._host = startup_args.host
        self._port = startup_args.port
        self._debug = startup_args.v
        self._num_processes = startup_args.p
        self._device_list = startup_args.device_list
        self._status = self.STATUS_LOADING
        # start webserver
        self._server = Bottle()
        # start translation workers
        #self._translator = Translator(self._models, self._num_processes, self._device_list)
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
        source_segments = [segment for segment in translation_request.segments]
        #todo: actual translation
        target_segments = [segment.upper() for segment in source_segments] # pseudo translation (to all caps)
        response_data = {
            'status': TranslationResponse.STATUS_OK,
            'segments': target_segments,
            'word_alignments': None,
            'word_probabilities': None,
        }
        translation_response = response_provider(self._style, **response_data)
        response.content_type = translation_response.get_content_type()
        return repr(translation_response)

    def start(self):
        """
        Starts the webserver.
        """
        self._route()
        self._server.run(host=self._host, port=self._port, debug=self._debug)

    def _route(self):
        """
        Routes webserver paths to functions.
        """
        self._server.route('/status', method="GET", callback=self.status)
        self._server.route('/translate', method="POST", callback=self.translate)


if __name__ == "__main__":
    parser = ConsoleInterfaceServer()
    startup_args = parser.parse_args()
    server = NematusServer(startup_args)
    server.start()
