#!/usr/bin/env python3

"""
Parses console arguments.
"""
import sys
import argparse
import uuid
import logging
from abc import ABCMeta

class BaseSettings(object, metaclass=ABCMeta):
    """
    All modes (abstract base class)
    """

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
        self._parser.add_argument(
            '-v', '--verbose', action="store_true",
            help="verbose mode")

        self._parser.add_argument(
            '-m', '--models', type=str, nargs='+', required=True,
            metavar="PATH",
            help="model to use; provide multiple models (with same " \
                 "vocabulary) for ensemble decoding")

        self._parser.add_argument(
            '-b', '--minibatch_size', type=int, default=80, metavar='INT',
            help="minibatch size (default: %(default)s)")

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
        for key, value in list(args.items()):
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

        if self._from_console_arguments:
            # don't open files if no console arguments are parsed
            self._parser.add_argument(
                '-i', '--input', type=argparse.FileType('r'),
                default=sys.stdin, metavar='PATH',
                help="input file (default: standard input)")

            self._parser.add_argument(
                '-o', '--output', type=argparse.FileType('w'),
                default=sys.stdout, metavar='PATH',
                help="output file (default: standard output)")

        self._parser.add_argument(
            '-k', '--beam_size', type=int, default=5, metavar='INT',
            help="beam size (default: %(default)s)")

        self._parser.add_argument(
            '-n', '--normalization_alpha', type=float, default=0.0, nargs="?",
            const=1.0, metavar="ALPHA",
            help="normalize scores by sentence length (with argument, " \
                 "exponentiate lengths by ALPHA)")

        # Support --n-best and --n_best (the dash version was added first, but
        # is inconsistent with the prevailing underscore style).
        group = self._parser.add_mutually_exclusive_group()
        group.add_argument(
            '--n_best', action="store_true",
            help="write n-best list (of size k)")
        group.add_argument(
            '--n-best', action="store_true",
            help=argparse.SUPPRESS)

        self._parser.add_argument(
            '--maxibatch_size', type=int, default=20, metavar='INT',
            help="size of maxibatch (number of minibatches that are sorted " \
                 "by length) (default: %(default)s)")

        self._parser.add_argument(
            '--sampling_temperature', type=float, default=1.0, nargs="?",
            const=1.0, metavar="FLOAT",
            help="softmax temperature used for sampling (default %(default)s)")

        self._parser.add_argument(
            '--translation_strategy', type=str, choices=['beam_search', 'sampling'], default="beam_search",
            help="translation_strategy, either beam_search or sampling (default: %(default)s)")

    def _set_additional_vars(self):
        self.request_id = uuid.uuid4()
        self.num_processes = 1

class ServerSettings(BaseSettings):
    """
    Console interface for server mode

    Most parameters required in default mode are provided with each translation
    request to the server (see `nematus/server/request.py`).
    """

    def _add_console_arguments(self):
        super(ServerSettings, self)._add_console_arguments()

        self._parser.add_argument(
            '--style', default='Nematus',
            help='API style; see `README.md` (default: Nematus)')

        self._parser.add_argument(
            '--host', default='0.0.0.0',
            help='host address (default: %(default)s)')

        self._parser.add_argument(
            '--port', type=int, default=8080, metavar='INT',
            help='host port (default: %(default)s)')

        self._parser.add_argument(
            '--threads', type=int, default=4, metavar='INT',
            help='number of threads (default: %(default)s)')

        self._parser.add_argument(
            '-p', '--num_processes', type=int, default=1, metavar='INT',
            help="number of processes (default: %(default)s)")


class ScorerBaseSettings(BaseSettings, metaclass=ABCMeta):
    """
    Base class for scorer and rescorer settings
    """

    def _add_console_arguments(self):
        super(ScorerBaseSettings, self)._add_console_arguments()

        self._parser.add_argument(
            '-n', '--normalization_alpha', type=float, default=0.0, nargs="?",
            const=1.0, metavar="ALPHA",
            help="normalize scores by sentence length (with argument, " \
                 "exponentiate lengths by ALPHA)")

        if self._from_console_arguments:
            # don't open files if no console arguments are parsed
            self._parser.add_argument(
                '-o', '--output', type=argparse.FileType('w'),
                default=sys.stdout, metavar='PATH',
                help="output file (default: standard output)")

            self._parser.add_argument(
                '-s', '--source', type=argparse.FileType('r'),
                required=True, metavar='PATH',
                help="source text file")


class ScorerSettings(ScorerBaseSettings):
    """
    Console interface for scoring (score.py)
    """
    def _add_console_arguments(self):
        super(ScorerSettings, self)._add_console_arguments()
        if self._from_console_arguments:
            self._parser.add_argument(
                '-t', '--target', type=argparse.FileType('r'), required=True,
                metavar='PATH',
                help="target text file")


class RescorerSettings(ScorerBaseSettings):
    """
    Console interface for rescoring (rescore.py)
    """
    def _add_console_arguments(self):
        super(RescorerSettings, self)._add_console_arguments()
        if self._from_console_arguments:
            self._parser.add_argument(
                '-i', '--input', type=argparse.FileType('r'),
                default=sys.stdin, metavar='PATH',
                help="input n-best list file (default: standard input)")
