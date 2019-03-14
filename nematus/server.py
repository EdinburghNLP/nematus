#!/usr/bin/env python3

"""
Runs Nematus as a Web Server.
"""

import json
import pkg_resources
import logging

from bottle import Bottle, request, response
from bottle_log import LoggingPlugin

from server.response import TranslationResponse
from server.api.provider import request_provider, response_provider
from settings import ServerSettings
from server_translator import Translator

class NematusServer(object):
    """
    Keeps a Nematus model in memory to answer http translation requests.
    """

    STATUS_LOADING = 'loading'
    STATUS_OK = 'ok'

    def __init__(self, server_settings):
        """
        Loads a translation model and initialises the webserver.

        @param server_settings: see `settings.py`
        """
        self._style = server_settings.style
        self._host = server_settings.host
        self._port = server_settings.port
        self._threads = server_settings.threads
        self._debug = server_settings.verbose
        self._models = server_settings.models
        self._num_processes = server_settings.num_processes
        self._status = self.STATUS_LOADING
        # start webserver
        self._server = Bottle()
        self._server.config['logging.level'] = 'DEBUG' if server_settings.verbose else 'WARNING'
        self._server.config['logging.format'] = '%(levelname)s: %(message)s'
        self._server.install(LoggingPlugin(self._server.config))
        logging.info("Starting Nematus Server")
        # start translation workers
        logging.info("Loading translation models")
        self._translator = Translator(server_settings)
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
        logging.debug("REQUEST - " + repr(translation_request))

        translations = self._translator.translate(
            translation_request.segments,
            translation_request.settings
        )
        response_data = {
            'status': TranslationResponse.STATUS_OK,
            'segments': [translation.target_words for translation in translations],
        }
        translation_response = response_provider(self._style, **response_data)
        logging.debug("RESPONSE - " + repr(translation_response))

        response.content_type = translation_response.get_content_type()
        return repr(translation_response)

    def start(self):
        """
        Starts the webserver.
        """
        self._route()
        self._server.run(host=self._host, port=self._port, debug=self._debug, server='tornado', threads=self._threads)
        self._cleanup()

    def _cleanup(self):
        """
        Graceful exit for components.
        """
        self._translator.shutdown()

    def _route(self):
        """
        Routes webserver paths to functions.
        """
        self._server.route('/status', method="GET", callback=self.status)
        self._server.route('/translate', method="POST", callback=self.translate)


if __name__ == "__main__":
    # parse console arguments
    server_settings = ServerSettings(from_console_arguments=True)
    server = NematusServer(server_settings)
    server.start()
