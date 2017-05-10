#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs Nematus as a Web Server.
"""

from bottle import Bottle, request, response
from server.response import TranslationResponse
from server.api.provider import request_provider, response_provider

class NematusServer(object):
    """
    Keeps a Nematus model in memory to answer http translation requests.
    """
    def __init__(self, model, style="Nematus", host="localhost", port=8080, debug=False):
        """
        Loads a translation model and initialises the webserver.

        @type model: str
        @param model: path to the Nematus translation model to be loaded
        @type style: str
        @param style: API style (see `README.md`)
        @type host: str
        @param host: the host address
        @type port: int
        @param port: the host port
        @type debug: bool
        @param debug: Debug mode (should be disabled in production)
        """
        self.style = style
        self.host = host
        self.port = port
        self.debug = debug
        self._server = Bottle()
        # load translation model
        #todo

    def translate(self):
        """
        Processes a translation request.
        """
        translation_request = request_provider(self.style, request)
        source_segments = [segment for segment in translation_request.segments]
        #todo: actual translation
        target_segments = [segment.upper() for segment in source_segments] # pseudo translation (to all caps)
        response_data = {
            'status': TranslationResponse.STATUS_OK,
            'segments': target_segments,
            'word_alignments': None,
            'word_probabilities': None,
        }
        translation_response = response_provider(self.style, **response_data)
        response.content_type = translation_response.get_content_type()
        return repr(translation_response)

    def start(self):
        """
        Starts the webserver.
        """
        self._route()
        self._server.run(host=self.host, port=self.port, debug=self.debug)

    def _route(self):
        """
        Routes webserver paths to functions.
        """
        self._server.route('/translate', method="POST", callback=self.translate)


if __name__ == "__main__":
    #todo: parse actual init params
    server = NematusServer('/foo/bar', debug=True)
    server.start()
