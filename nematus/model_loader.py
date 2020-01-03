import json
import logging
import os
import re
import sys

import numpy
import tensorflow as tf
from tensorflow.python.framework import ops
#import tensorflow.contrib.slim as slim #tensorflow.contrib.framework ???

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from .exponential_smoothing import ExponentialSmoothing
    from . import training_progress
except (ModuleNotFoundError, ImportError) as e:
    from exponential_smoothing import ExponentialSmoothing
    import training_progress

def init_or_restore_variables(config, sess, ensemble_scope=None, train=False):
    """Initializes all variables or restores variables from a checkpoint.

    Prior to calling this function, the model (or models if using an ensemble)
    should already have been created. If a model uses exponential smoothing,
    then the _smooth versions of its variables should also have been created
    (by constructing an ExponentialSmoothing object). This function will then
    initialize the variables or restore them from a checkpoint (if one can be
    found).

    When using an ensemble, this function should be called once for each
    model, with each call using model-specific config and ensemble_scope
    arguments.

    Args:
      config: Namespace object specifying the config for the current model.
      sess: a TensorFlow session.
      ensemble_scope: a tf.variable_scope for the current model if ensembling.
      train: Boolean specifying that the model is being used for training.

    Returns:
      If train is True, returns a pair (saver, progress) where saver is a
      tf.train.Saver and progress is a training_progress.TrainingProgress
      object. Otherwise, just returns the saver.
    """

    accum_regex = re.compile('^accum\d+$')

    def is_excluded_variable(name):
        # Exclude gradient accumulation variables.
        if accum_regex.match(name):
            return True
        if name == 'accumulated_loss':
            return True
        return False

    variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    # Construct a mapping between saved variable names and names in the current
    # scope. There are two reasons why names might be different:
    #
    #   1. This model is part of an ensemble, in which case a model-specific
    #       name scope will be active.
    #
    #   2. The saved model is from an old version of Nematus (before deep model
    #        support was added) and uses a different variable naming scheme
    #        for the GRUs.

    var_map = {}
    for v in variables:
        name = v.name.split(':')[0]
        if ensemble_scope == None:
            saved_name = name
        elif v.name.startswith(ensemble_scope.name + "/"):
            saved_name = name[len(ensemble_scope.name)+1:]
            # The ensemble scope is repeated for Adam variables. See
            # https://github.com/tensorflow/tensorflow/issues/8120
            if saved_name.startswith(ensemble_scope.name + "/"):
                saved_name = saved_name[len(ensemble_scope.name)+1:]
        else: # v belongs to a different model in the ensemble.
            continue
        if is_excluded_variable(saved_name):
            continue
        if config.model_version == 0.1:
            # Backwards compatibility with the old variable naming scheme.
            saved_name = _revert_variable_name(saved_name, 0.1)
        var_map[saved_name] = v
    saver = tf.compat.v1.train.Saver(var_map, max_to_keep=None)

    # compute reload model filename
    reload_filename = None
    if config.reload == 'latest_checkpoint':
        checkpoint_dir = os.path.dirname(config.saveto)
        reload_filename = tf.train.latest_checkpoint(checkpoint_dir)
        if reload_filename != None:
            if (os.path.basename(reload_filename).rsplit('-', 1)[0] !=
                os.path.basename(config.saveto)):
                logging.error("Mismatching model filename found in the same directory while reloading from the latest checkpoint")
                sys.exit(1)
            logging.info('Latest checkpoint found in directory ' + os.path.abspath(checkpoint_dir))
    elif config.reload != None:
        reload_filename = config.reload
    if (reload_filename == None) and (config.prior_model != None):
        logging.info('Initializing model parameters from prior')
        reload_filename = config.prior_model

    # initialize or reload training progress
    if train:
        progress = training_progress.TrainingProgress()
        progress.bad_counter = 0
        progress.uidx = 0
        progress.eidx = 0
        progress.estop = False
        progress.history_errs = []
        progress.valid_script_scores = []
        if reload_filename and config.reload_training_progress:
            path = reload_filename + '.progress.json'
            if os.path.exists(path):
                logging.info('Reloading training progress')
                progress.load_from_json(path)
                if (progress.estop == True or
                    progress.eidx > config.max_epochs or
                    progress.uidx >= config.finish_after):
                    logging.warning('Training is already complete. Disable reloading of training progress (--no_reload_training_progress) or remove or modify progress file (%s) to train anyway.' % path)
                    sys.exit(0)

    # load prior model
    if train and config.prior_model != None:
        load_prior(config, sess, saver)

    init_op = tf.compat.v1.global_variables_initializer()

    # initialize or restore model
    if reload_filename == None:
        logging.info('Initializing model parameters from scratch...')
        sess.run(init_op)
    else:
        logging.info('Loading model parameters from file ' + os.path.abspath(reload_filename))
        if train:
            # Initialize all variables before restoring from the checkpoint.
            # This is to allow for variables that are not saved to the
            # checkpoint. Currently that is just the gradient accumulation
            # variables, which are unusual in that they persist across multiple
            # sessions during training (and therefore need to be variables) but
            # are regularly reset to zero.
            sess.run(init_op)
        saver.restore(sess, os.path.abspath(reload_filename))
    logging.info('Done')

    if train:
        return saver, progress
    else:
        return saver


def load_prior(config, sess, saver):
     logging.info('Loading prior model parameters from file ' + os.path.abspath(config.prior_model))
     saver.restore(sess, os.path.abspath(config.prior_model))

     # fill prior variables with the loaded values
     prior_variables = tf.compat.v1.get_collection_ref('prior_variables')
     prior_variables_dict = dict([(v.name, v) for v in prior_variables])
     assign_tensors = []
     with tf.compat.v1.variable_scope('prior'):
         for v in tf.compat.v1.trainable_variables():
             prior_name = 'loss/prior/'+v.name
             prior_variable = prior_variables_dict[prior_name]
             assign_tensors.append(prior_variable.assign(v))
     tf.compat.v1.variables_initializer(prior_variables)
     sess.run(assign_tensors)


# for backwards compatibility with old models
def _revert_variable_name(name, old_version):
    assert old_version == 0.1
    if name.endswith("/Adam"):
        prefix = name[:-len("/Adam")]
        return _revert_variable_name(prefix, old_version) + "/Adam"
    if name.endswith("/Adam_1"):
        prefix = name[:-len("/Adam_1")]
        return _revert_variable_name(prefix, old_version) + "/Adam_1"
    if "forward-stack/level0/gru0" in name:
        return name.replace("forward-stack/level0/gru0", "forwardEncoder")
    if "backward-stack/level0/gru0" in name:
        return name.replace("backward-stack/level0/gru0", "backwardEncoder")
    if "decoder/base/gru0" in name:
        return name.replace("decoder/base/gru0", "decoder")
    if "decoder/base/attention" in name:
        return name.replace("decoder/base/attention", "decoder")
    if "decoder/base/gru1" in name:
        tmp = name.replace("decoder/base/gru1", "decoder")
        if tmp.endswith("/new_mean"):
            return tmp.replace("/new_mean", "_1/new_mean")
        elif tmp.endswith("/new_std"):
            return tmp.replace("/new_std", "_1/new_std")
        else:
            return tmp + "_1"
    if "decoder/embedding" in name:
        return name.replace("decoder/embedding", "decoder/y_embeddings_layer")
    return name
