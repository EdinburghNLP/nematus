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