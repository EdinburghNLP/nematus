'''
Training progress
'''

import json

class TrainingProgress(object):
    '''
    Object used to store, serialize and deserialize pure python variables that change during training and should be preserved in order to properly restart the training process
    '''

    def load_from_json(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as fh:
            self.__dict__.update(json.load(fh))

    def save_to_json(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as fh:
            # TODO ensure_ascii=False?
            json.dump(self.__dict__, fh, indent=2)
