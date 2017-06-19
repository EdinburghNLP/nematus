#!/usr/bin/env python

"""
Parses console arguments.
"""
import sys
import argparse
from abc import ABCMeta, abstractmethod

from settings import DecoderSettings, TranslationSettings, ServerSettings

class ConsoleInterface(object):
    """
    All modes (abstract base class)
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._add_shared_arguments()
        self._add_arguments()

    def _add_shared_arguments(self):
        """
        Console arguments used in all modes
        """
        self._parser.add_argument('--models', '-m', type=str, nargs = '+', required=True, metavar="MODEL",
                                  help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
        self._parser.add_argument('-p', type=int, default=1,
                                  help="Number of processes (default: %(default)s))")
        self._parser.add_argument('--device-list', '-dl', type=str, nargs='*', required=False, metavar="DEVICE",
                                  help="User specified device list for multi-thread decoding (default: [])")
        self._parser.add_argument('-v', action="store_true", help="verbose mode.")

    @abstractmethod
    def _add_arguments(self):
        """
        Console arguments used in specific mode
        """
        pass # to be implemented in subclass

    def parse_args(self):
        """
        Returns the parsed console arguments
        """
        return self._parser.parse_args()

    def get_decoder_settings(self):
        """
        Returns a `DecoderSettings` object based on the parsed console
        arguments.
        """
        args = self.parse_args()
        return DecoderSettings(args)

class ConsoleInterfaceDefault(ConsoleInterface):
    """
    Console interface for default mode
    """

    def _add_arguments(self):
        self._parser.add_argument('-k', type=int, default=5,
                                  help="Beam size (default: %(default)s))")
        self._parser.add_argument('-n', action="store_true",
                                  help="Normalize scores by sentence length")
        self._parser.add_argument('-c', action="store_true", help="Character-level")
        self._parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                                  default=sys.stdin, metavar='PATH',
                                  help="Input file (default: standard input)")
        self._parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                                  default=sys.stdout, metavar='PATH',
                                  help="Output file (default: standard output)")
        self._parser.add_argument('--output_alignment', '-a', type=argparse.FileType('w'),
                                  default=None, metavar='PATH',
                                  help="Output file for alignment weights (default: standard output)")
        self._parser.add_argument('--json_alignment', action="store_true",
                                  help="Output alignment in json format")
        self._parser.add_argument('--n-best', action="store_true",
                                  help="Write n-best list (of size k)")
        self._parser.add_argument('--suppress-unk', action="store_true",
                                  help="Suppress hypotheses containing UNK.")
        self._parser.add_argument('--print-word-probabilities', '-wp',
                                  action="store_true", help="Print probabilities of each word")
        self._parser.add_argument('--search_graph', '-sg',
                                  help="Output file for search graph rendered as PNG image")

    def get_translation_settings(self):
        """
        Returns a `TranslationSettings` object based on the parsed console
        arguments.
        """
        args = self.parse_args()
        return TranslationSettings(args)


class ConsoleInterfaceServer(ConsoleInterface):
    """
    Console interface for server mode

    Most parameters required in default mode are provided with each translation
    request to the server (see `nematus/server/request.py`).
    """

    def _add_arguments(self):
        self._parser.add_argument('--style', default='Nematus',
                                  help='API style; see `README.md` (default: Nematus)')
        self._parser.add_argument('--host', default='localhost',
                                  help='Host address (default: localhost)')
        self._parser.add_argument('--port', type=int, default=8080,
                                  help='Host port (default: 8080)')

    def get_server_settings(self):
        """
        Returns a `ServerSettings` object based on the parsed console
        arguments.
        """
        args = self.parse_args()
        return ServerSettings(args)
