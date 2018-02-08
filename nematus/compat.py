'''
Default options for backward compatibility
'''

#hacks for using old models with missing options (dict is modified in-place)

from __future__ import unicode_literals

def fill_options(options):

    # old theano versions
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
        options['dim_per_factor'] = [options['dim_word']]
    if not 'model_version' in options:
        options['model_version'] = 0
    if not 'tie_decoder_embeddings' in options:
        options['tie_decoder_embeddings'] = False
    if not 'map_decay_c' in options:
        options['map_decay_c'] = 0.0

    # name changes in TF
    if not 'source_vocab_size' in options:
        options['source_vocab_size'] = options['n_words_src']
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
    if not 'source_vocab' in options:
        options['source_vocab'] = options['dictionaries'][0]
        options['target_vocab'] = options['dictionaries'][1]

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
