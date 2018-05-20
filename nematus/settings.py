#!/usr/bin/env python

"""
Parses console arguments.
"""
import sys
import argparse
import uuid
import logging
from abc import ABCMeta

class BaseSettings(object):
    """
    All modes (abstract base class)
    """
    __metaclass__ = ABCMeta

    def __init__(self, from_console_arguments=False):
        self._from_console_arguments = from_console_arguments
        self._parser = argparse.ArgumentParser()
        self._add_console_arguments()
        self._set_console_arguments()
        self._set_additional_vars()

    def _add_console_arguments(self):
        """
        Console arguments used in all modes
        """
        self._parser.add_argument('--models', '-m', type=str, nargs = '+', required=True, metavar="MODEL",
                                  help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
        self._parser.add_argument('-p', dest='num_processes', type=int, default=1,
                                  help="Number of processes (default: %(default)s))")
        self._parser.add_argument('-v', dest='verbose', action="store_true", help="verbose mode.")

    def _set_console_arguments(self):
        """
        Parses console arguments and loads them into the namespace of this
        object.

        If there are no console arguments, the argument parser's default values
        (see `self._parse_shared_console_arguments` and
        `self._parse_individual_console_arguments`) are used.
        """
        if self._from_console_arguments:
            args = vars(self._parser.parse_args())
        else:
            args = {a.dest: self._parser.get_default(a.dest) for a in self._parser._actions}
        for key, value in args.items():
            setattr(self, key, value)

    def _set_additional_vars(self):
        """
        Adds additional variables/constants to this object. They can be derived
        or independent from parsed console arguments.
        """
        pass # override in subclass


class TranslationSettings(BaseSettings):
    """
    Console interface for file translation mode
    """

    def _add_console_arguments(self):
        super(TranslationSettings, self)._add_console_arguments()

        self._parser.add_argument('-k', dest='beam_width', type=int, default=5,
                                  help="Beam size (default: %(default)s))")
        self._parser.add_argument('-n', dest='normalization_alpha', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                                  help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")
        self._parser.add_argument('-c', dest='char_level', action="store_true", help="Character-level")

        if self._from_console_arguments: # don't open files if no console arguments are parsed
            self._parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                                      default=sys.stdin, metavar='PATH',
                                      help="Input file (default: standard input)")
            self._parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                                      default=sys.stdout, metavar='PATH',
                                      help="Output file (default: standard output)")

        self._parser.add_argument('--n-best', action="store_true",
                                  help="Write n-best list (of size k)")

    def _set_additional_vars(self):
        self.request_id = uuid.uuid4()

class ServerSettings(BaseSettings):
    """
    Console interface for server mode

    Most parameters required in default mode are provided with each translation
    request to the server (see `nematus/server/request.py`).
    """

    def _add_console_arguments(self):
        super(ServerSettings, self)._add_console_arguments()
        self._parser.add_argument('--style', default='Nematus',
                                  help='API style; see `README.md` (default: Nematus)')
        self._parser.add_argument('--host', default='0.0.0.0',
                                  help='Host address (default: 0.0.0.0)')
        self._parser.add_argument('--port', type=int, default=8080,
                                  help='Host port (default: 8080)')
        self._parser.add_argument('--threads', type=int, default=4,
                                  help='Number of threads (default: 4)')


class ScorerBaseSettings(BaseSettings):
    """
    Base class for scorer and rescorer settings
    """
    __metaclass__ = ABCMeta

    def _add_console_arguments(self):
        super(ScorerBaseSettings, self)._add_console_arguments()
        self._parser.add_argument('-b', type=int, default=80,
                                  help="Minibatch size (default: %(default)s))")
        self._parser.add_argument('-n', dest='normalization_alpha', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                                  help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")
        if self._from_console_arguments: # don't open files if no console arguments are parsed
            self._parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                                      default=sys.stdout, metavar='PATH', help="Output file (default: standard output)")
            self._parser.add_argument('--source', '-s', type=argparse.FileType('r'),
                                      required=True, metavar='PATH', help="Source text file")


class ScorerSettings(ScorerBaseSettings):
    """
    Console interface for scoring (score.py)
    """
    def _add_console_arguments(self):
        super(ScorerSettings, self)._add_console_arguments()
        if self._from_console_arguments:
            self._parser.add_argument('--target', '-t', type=argparse.FileType('r'),
                                      required=True, metavar='PATH', help="Target text file")


class RescorerSettings(ScorerBaseSettings):
    """
    Console interface for rescoring (rescore.py)
    """
    def _add_console_arguments(self):
        super(RescorerSettings, self)._add_console_arguments()
        if self._from_console_arguments:
            self._parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                                      default=sys.stdin, metavar='PATH', help="Input n-best list file (default: standard input)")
