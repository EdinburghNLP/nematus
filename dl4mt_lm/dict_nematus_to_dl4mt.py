#!/usr/bin/python

"""
Convert Nematus vocabulary dictionary (json) to dl4mt format (pickle)
"""

import sys
import pickle
import operator
import json

from collections import OrderedDict


if '.json' not in sys.argv[1]:
    print('USAGE: dict_nematus_to_dl4mt dictionary.json')
    exit(1)

dic_name = sys.argv[1].replace('.json', '')
nematus_dict = json.load(open(dic_name+'.json', 'r'))
nematus_dict = dict((key.encode("utf-8"), value) for (key,value) in nematus_dict.items())
dl4mt_dict = OrderedDict(sorted(nematus_dict.items(), key=operator.itemgetter(1)))
pickle.dump(dl4mt_dict, open(dic_name+'.pkl', 'w'))
