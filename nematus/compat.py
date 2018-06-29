'''
Default options for backward compatibility
'''

#hacks for using old models with missing options (dict is modified in-place)

from __future__ import unicode_literals

def fill_options(options):

    # does this look like an old Theano config?
    from_theano_version = ('embedding_size' not in options)

    # name changes in TF
    if not 'source_vocab_sizes' in options:
        if 'source_vocab_size' in options:
            first_factor_size = options['source_vocab_size']
        else:
            first_factor_size = options['n_words_src']
        num_factors = options.get('factors', 1)
        options['source_vocab_sizes'] = [first_factor_size] * num_factors
    if not 'target_vocab_size' in options:
        options['target_vocab_size'] = options['n_words']
    if not 'embedding_size' in options:
        options['embedding_size'] = options['dim_word']
    if not 'use_layer_norm' in options:
        if 'layer_normalisation' in options:
            options['use_layer_norm'] = options['layer_normalisation']
        else:
            options['use_layer_norm'] = False
    if not 'state_size' in options:
        options['state_size'] = options['dim']

    # handle vocab dictionaries
    options['source_dicts'] = options['dictionaries'][:-1]
    options['target_dict'] = options['dictionaries'][-1]

    # set defaults for newer options that may not be present
    if not 'use_dropout' in options:
        options['use_dropout'] = False
    if not 'dropout_embedding' in options:
        options['dropout_embedding'] = 0
    if not 'dropout_hidden' in options:
        options['dropout_hidden'] = 0
    if not 'dropout_source' in options:
        options['dropout_source'] = 0
    if not 'dropout_target' in options:
        options['dropout_target'] = 0
    if not 'factors' in options:
        options['factors'] = 1
    if not 'dim_per_factor' in options:
        options['dim_per_factor'] = [options['embedding_size']]
    if not 'tie_decoder_embeddings' in options:
        options['tie_decoder_embeddings'] = False
    if not 'map_decay_c' in options:
        options['map_decay_c'] = 0.0
    if not 'enc_depth' in options:
        options['enc_depth'] = 1
    if not 'enc_recurrence_transition_depth' in options:
        options['enc_recurrence_transition_depth'] = 1
    if not 'dec_depth' in options:
        options['dec_depth'] = 1
    if not 'dec_base_recurrence_transition_depth' in options:
        options['dec_base_recurrence_transition_depth'] = 2
    if not 'dec_high_recurrence_transition_depth' in options:
        options['dec_high_recurrence_transition_depth'] = 1
    if not 'dec_deep_context' in options:
        options['dec_deep_context'] = False

    # set the default model version.
    if not 'model_version' in options:
        if from_theano_version and options['use_dropout']:
            assert False # version 0 dropout is not supported in TensorFlow Nematus
        else:
            options['model_version'] = 0.1

    # extra config options in TF; only translation_maxlen matters for translation/scoring
    if not 'translation_maxlen' in options:
        options['translation_maxlen'] = 200
    if not 'optimizer' in options:
        options['optimizer'] = 'adam'
    if not 'learning_rate' in options:
        options['learning_rate'] = 0.0001
    if not 'clip_c' in options:
        options['clip_c'] = 1.
    if not 'output_hidden_activation' in options:
        options['output_hidden_activation'] = 'tanh'


# for backwards compatibility with old models
def revert_variable_name(name, old_version):
    assert old_version == 0.1
    if name.endswith("/Adam"):
        return revert_variable_name(name[:-len("/Adam")], old_version) + "/Adam"
    if name.endswith("/Adam_1"):
        return revert_variable_name(name[:-len("/Adam_1")], old_version) + "/Adam_1"
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
