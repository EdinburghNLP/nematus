#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration containers.
"""

import uuid

class DecoderSettings(object):

    def __init__(self, parsed_console_arguments=None):
        """
        Decoder settings are initialised with default values, unless parsed
        console arguments as returned by a `ConsoleInterface`'s `parse_args()`
        method are provided.
        """
        self.models = []
        self.num_processes = 1
        self.device_list = []
        self.verbose = False
        if parsed_console_arguments:
            self.update_from(parsed_console_arguments)

    def update_from(self, parsed_console_arguments):
        """
        Updates decoder settings based on @param parsed_console_arguments,
        as returned by a `ConsoleInterface`'s `parse_args()` method.
        """
        args = parsed_console_arguments
        self.models = args.models
        self.num_processes = args.p
        self.device_list = args.device_list
        self.verbose = args.v


class TranslationSettings(object):

    ALIGNMENT_TEXT = 1
    ALIGNMENT_JSON = 2

    def __init__(self, parsed_console_arguments=None):
        """
        Translation settings are initialised with default values, unless parsed
        console arguments as returned by a `ConsoleInterface`'s `parse_args()`
        method are provided.
        """
        self.request_id = uuid.uuid4()
        self.beam_width = 5
        self.normalize = False
        self.char_level = False
        self.n_best = 1
        self.suppress_unk = False
        self.get_word_probs = False
        self.get_alignment = False
        self.alignment_type = None
        self.alignment_filename = None
        self.get_search_graph = False
        self.search_graph_filename = None
        if parsed_console_arguments:
            self.update_from(parsed_console_arguments)

    def update_from(self, parsed_console_arguments):
        """
        Updates translation settings based on @param parsed_console_arguments,
        as returned by a `ConsoleInterface`'s `parse_args()` method.
        """
        args = parsed_console_arguments
        self.beam_width = args.k
        self.normalize = args.n
        self.char_level = args.c
        self.n_best = args.n_best
        self.suppress_unk = args.suppress_unk
        self.get_word_probs = args.print_word_probabilities
        if args.output_alignment:
            self.get_alignment = True
            self.alignment_filename = args.output_alignment
            if args.json_alignment:
                self.alignment_type = self.ALIGNMENT_JSON
            else:
                self.alignment_type = self.ALIGNMENT_TEXT
        else:
            self.get_alignment = False
        if args.search_graph:
            self.get_search_graph = True
            self.search_graph_filename = args.search_graph
        else:
            self.get_search_graph = False
            self.search_graph_filename = None


class ServerSettings(object):

    def __init__(self, parsed_console_arguments=None):
        """
        Server settings are initialised with default values, unless parsed
        console arguments as returned by a `ConsoleInterface`'s `parse_args()`
        method are provided.
        """
        self.style = "Nematus" #TODO: use constant
        self.host = "localhost"
        self.port = 8080
        if parsed_console_arguments:
            self.update_from(parsed_console_arguments)

    def update_from(self, parsed_console_arguments):
        """
        Updates decoder settings based on @param parsed_console_arguments,
        as returned by a `ConsoleInterface`'s `parse_args()` method.
        """
        args = parsed_console_arguments
        self.style = args.style
        self.host = args.host
        self.port = args.port
