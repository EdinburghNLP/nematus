#!/usr/bin/env python

"""
Implements providors for TranslationRequest and TranslationResponse objects
of a specific API style.
"""

def request_provider(style, request):
    """
    Turns a raw request body into a TranslationRequest of a given API style
    @param style.
    """
    from .nematus_style import TranslationRequestNematus
    mapping = {
        'Nematus': TranslationRequestNematus
    }
    try:
        return mapping[style](request)
    except IndexError:
        raise NotImplementedError("Invalid API style: {0}".format(style))

def response_provider(style, **response_args):
    """
    Formats @param response_args as a TranslationResponse of a given API style
    @param style.
    """
    from .nematus_style import TranslationResponseNematus
    mapping = {
        'Nematus': TranslationResponseNematus
    }
    try:
        return mapping[style](**response_args)
    except IndexError:
        raise NotImplementedError("Invalid API style: {0}".format(style))
