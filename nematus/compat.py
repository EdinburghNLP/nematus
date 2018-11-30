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
    if not 'theano_compat' in options:
        options['theano_compat'] = from_theano_version
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
    if not 'tie_encoder_decoder_embeddings' in options:
        options['tie_encoder_decoder_embeddings'] = False
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
    if not 'target_embedding_size' in options:
        if options['tie_encoder_decoder_embeddings'] == True:
            options['target_embedding_size'] = options['dim_per_factor'][0]
        else:
            options['target_embedding_size'] = options['embedding_size']
    if not 'label_smoothing' in options:
        options['label_smoothing'] = 0.0
    if not 'softmax_mixture_size' in options:
        options['softmax_mixture_size'] = 1
    if not 'adam_beta1' in options:
        options['adam_beta1'] = 0.9
    if not 'adam_beta2' in options:
        options['adam_beta2'] = 0.999
    if not 'adam_epsilon' in options:
        options['adam_epsilon'] = 1e-08

    # Nematode compatibility
    if not 'translation_max_len' in options:
        options['translation_max_len'] = options['translation_maxlen']
    if not 'num_encoder_layers' in options:
        options['num_encoder_layers'] = options['transformer_encoder_layers']
    if not 'num_decoder_layers' in options:
        options['num_decoder_layers'] = options['transformer_decoder_layers']
    if not 'ffn_hidden_size' in options:
        options['ffn_hidden_size'] = options['transformer_ffn_hidden_size']
    # TODO general?
    if not 'num_heads' in options:
        options['num_heads'] = options['transformer_num_heads']
    if not 'dropout_embeddings' in options:
        options['dropout_embeddings'] = options['transformer_dropout_embeddings']
    if not 'dropout_residual' in options:
        options['dropout_residual'] = options['transformer_dropout_residual']
    if not 'dropout_relu' in options:
        options['dropout_relu'] = options['transformer_dropout_relu']
    if not 'dropout_attn' in options:
        options['dropout_attn'] = options['transformer_dropout_attn']

    # set the default model version.
    if not 'model_version' in options:
        if from_theano_version and options['use_dropout']:
            assert False # version 0 dropout is not supported in TensorFlow Nematus
        else:
            options['model_version'] = 0.1

    # extra config options in TF; only translation_maxlen matters for translation/scoring
    if not 'translation_maxlen' in options:
        options['translation_maxlen'] = 200
    options['optimizer'] = 'adam'
    if not 'learning_rate' in options:
        options['learning_rate'] = 0.0001
    if not 'clip_c' in options:
        options['clip_c'] = 1.
    if not 'output_hidden_activation' in options:
        options['output_hidden_activation'] = 'tanh'


def create_nematode_config_or_die(config):
    # TODO Check that config is compatible with Nematode
    nematode_config = config
    nematode_config.hidden_size = config.state_size
    nematode_config.label_smoothing_discount = config.label_smoothing
    nematode_config.translation_max_len = config.translation_maxlen
    nematode_config.num_encoder_layers = config.transformer_encoder_layers
    nematode_config.num_decoder_layers = config.transformer_decoder_layers
    nematode_config.ffn_hidden_size = config.transformer_ffn_hidden_size
    nematode_config.num_heads = config.transformer_num_heads
    nematode_config.dropout_embeddings = config.transformer_dropout_embeddings
    nematode_config.dropout_residual = config.transformer_dropout_residual
    nematode_config.dropout_relu = config.transformer_dropout_relu
    nematode_config.dropout_attn = config.transformer_dropout_attn
    return nematode_config


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
