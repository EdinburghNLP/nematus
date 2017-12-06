#!/usr/bin/python

"""
Convert Nematus vocabulary dictionary (json) to dl4mt format
(pickle).

This is necessary to train a "deep fusion" model in Nematus.
The MT and LM vocabularies are identical.
"""


import sys
import pickle
import operator
import json

from collections import OrderedDict


if '.json' not in sys.argv[1]:
    print('USAGE: dict_nematus_to_dl4mt.py dictionary.json')
    exit(1)

dict_name = sys.argv[1].replace('.json', '')
nematus_dict = json.load(open(dict_name+'.json', 'r'))
dl4mt_dict = OrderedDict(
    sorted(((key.encode("utf-8"), value) for (key,value) in nematus_dict.items()), key=operator.itemgetter(1))
    )

pickle.dump(dl4mt_dict, open(dict_name+'.pkl', 'w'))
