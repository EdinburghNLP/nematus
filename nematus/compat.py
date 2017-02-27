'''
Default options for backward compatibility
'''

#hacks for using old models with missing options (dict is modified in-place)
def fill_options(options):
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
    if not 'tie_encoder_decoder_embeddings' in options:
        options['tie_encoder_decoder_embeddings'] = False
    if not 'tie_decoder_embeddings' in options:
        options['tie_decoder_embeddings'] = False
    if not 'enc_depth' in options:
        options['enc_depth'] = 1
    if not 'dec_depth' in options:
        options['dec_depth'] = 1
    if not 'dec_base_recurrence_transition_depth' in options:
        options['dec_base_recurrence_transition_depth'] = 2
    if not 'dec_deep_context' in options:
        if 'deep_include_ctx' in options:
           options['dec_deep_context'] = options['deep_include_ctx']
        else:
            options['deep_include_ctx'] = False
    if not 'encoder_truncate_gradient' in options:
        options['encoder_truncate_gradient'] = -1
    if not 'decoder_truncate_gradient' in options:
        options['decoder_truncate_gradient'] = -1
    if not 'enc_depth_bidirectional' in options:
        options['enc_depth_bidirectional'] = options['enc_depth']
    if not 'output_depth' in options:
        options['output_depth'] = 1
    if not 'output_depth' in options:
        options['output_depth'] = 1
