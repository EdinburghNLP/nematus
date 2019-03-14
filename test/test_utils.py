#!/usr/bin/env python3

import sys
import os
import requests
from shutil import copyfile

sys.path.append(os.path.abspath('../nematus'))
from theano_tf_convert import theano_to_tensorflow_model

def load_wmt16_model(src, target):
        path = os.path.join('models', '{0}-{1}'.format(src,target))
        try:
            os.makedirs(path)
        except OSError:
            pass
        for filename in ['model.npz.json', 'model.npz', 'vocab.{0}.json'.format(src), 'vocab.{0}.json'.format(target)]:
            if not os.path.exists(os.path.join(path, filename)):
                if filename == 'model.npz' and os.path.exists(os.path.join(path, 'model.npz.index')):
                    continue
                r = requests.get('http://data.statmt.org/rsennrich/wmt16_systems/{0}-{1}/'.format(src,target) + filename, stream=True)
                with open(os.path.join(path, filename), 'wb') as f:
                    for chunk in r.iter_content(1024**2):
                        f.write(chunk)

                # regression test is based on Theano model - convert to TF names
                if filename == 'model.npz.json' and not os.path.exists(os.path.join(path, 'model.npz.index')):
                    copyfile(os.path.join(path, 'model.npz.json'), os.path.join(path, 'model-theano.npz.json'))
                elif filename == 'model.npz' and not os.path.exists(os.path.join(path, 'model.npz.index')):
                    os.rename(os.path.join(path, 'model.npz'), os.path.join(path, 'model-theano.npz'))
                    theano_to_tensorflow_model(os.path.join(path, 'model-theano.npz'), os.path.join(path, 'model.npz'))
