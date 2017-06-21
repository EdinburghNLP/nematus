#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import requests # use `pip install requests` if not available on your system

SOURCE_SEGMENTS = {
    "de":"Die Wahrheit ist selten rein und nie einfach .".split(),
    "en":"The truth is rarely pure and never simple .".split()
}

class Client(object):
    """
    A sample client for Nematus Server instances.

    Uses the Nematus API style, i.e., the server (`server.py`) must be started
    with `style=Nematus` to serve requests from this client.
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.headers = {
            'content-type': 'application/json'
        }

    def _get_url(self, path='/'):
        return "http://{0}:{1}{2}".format(self.host, self.port, path)

    def translate(self, segment):
        """
        Returns the translation of a list of segments.
        """
        return self.translate_segments([segment])[0]

    def translate_segments(self, segments):
        """
        Returns the translation of a single segment.
        """
        payload = json.dumps({'segments': segments})
        url = self._get_url('/translate')
        response = requests.post(url, headers=self.headers, data=payload)
        return [segment['translation'] for segment in response.json()['data']]

    def print_server_status(self):
        """
        Prints the server's status report.
        """
        url = self._get_url('/status')
        response = requests.get(url, headers=self.headers)
        print json.dumps(response.json(), indent=4)


if __name__ == "__main__":
    host = 'localhost'
    port = 8080
    client = Client(host, port)
    client.print_server_status()
    source_segment = SOURCE_SEGMENTS['de']
    print 'Translating "{0}"'.format(source_segment)
    target_segment = client.translate(source_segment)
    print target_segment
