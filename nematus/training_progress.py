'''
Training progress
'''

import sys
import json

import util

class TrainingProgress(object):
    '''
    Object used to store, serialize and deserialize pure python variables that change during training and should be preserved in order to properly restart the training process
    '''

    def load_from_json(self, file_name):
        self.__dict__.update(json.load(open(file_name, 'r', encoding="utf8")))

    def save_to_json(self, file_name):
        json.dump(self.__dict__, open(file_name, 'w'), indent=2)
