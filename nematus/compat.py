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
    if not 'encoder_truncate_gradient' in options:
        options['encoder_truncate_gradient'] = -1
    if not 'decoder_truncate_gradient' in options:
        options['decoder_truncate_gradient'] = -1
    if not 'reload_training_progress' in options:
        options['reload_training_progress'] = True
    if not 'use_domain_interpolation' in options:
        options['use_domain_interpolation'] = False
    if not 'domain_interpolation_min' in options:
        options['decoder_truncate_gradient'] = 0.1
    if not 'domain_interpolation_max' in options:
        options['decoder_truncate_gradient'] = 1.0
    if not 'domain_interpolation_inc' in options:
        options['decoder_truncate_gradient'] = 0.1
    if not 'domain_interpolation_indomain_datasets' in options:
        options['domain_interpolation_indomain_datasets'] = ['indomain.en', 'indomain.fr']




