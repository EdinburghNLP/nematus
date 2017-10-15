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

    if not 'enc_recurrence_transition_depth' in options:
        options['enc_recurrence_transition_depth'] = 1
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
    if not 'decoder' in options:
        options['decoder'] = 'gru_cond'
    if not 'decoder_deep' in options:
        options['decoder_deep'] = 'gru'
    if not 'decoder_gru_no_reset_gate' in options:
        options['decoder_gru_no_reset_gate'] = False
    if not 'layer_normalisation' in options:
        options['layer_normalisation'] = False
    if not 'weight_normalisation' in options:
        options['weight_normalisation'] = False
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

    if not 'dec_high_recurrence_transition_depth' in options:
        if options['decoder_deep'] == 'gru_cond':
            options['dec_high_recurrence_transition_depth'] = 2
        else:
            options['dec_high_recurrence_transition_depth'] = 1

    if not 'attention_hidden_activation' in options:
        options['attention_hidden_activation'] = 'tanh'
    if not 'attention_hidden_dim' in options:
        options['attention_hidden_dim'] = -1
    if not 'output_hidden_activation' in options:
        options['output_hidden_activation'] = 'tanh'
    if not 'output_crelu_hidden_dim' in options:
        options['output_crelu_hidden_dim'] = -1
    if not 'decoder_initial_state_hidden_activation' in options:
        options['decoder_initial_state_hidden_activation'] = 'tanh'
    if not 'decoder_initial_state_crelu_hidden_dim' in options:
        options['decoder_initial_state_crelu_hidden_dim'] = -1
    if not 'decoder_initial_state_fixed' in options:
        options['decoder_initial_state_fixed'] = False

    if not 'decoder_main_activation' in options:
        options['decoder_main_activation'] = 'tanh'
    if not 'decoder_main_recurrent_identity_init' in options:
        options['decoder_main_recurrent_identity_init'] = False
    if not 'decoder_post_activation_input' in options:
        options['decoder_post_activation_input'] = False
    if not 'decoder_zero_init_main_input' in options:
        options['decoder_zero_init_main_input'] = False
    if not 'decoder_crelurhn_cond_no_layer_norm_on_state' in options:
        options['decoder_crelurhn_cond_no_layer_norm_on_state'] = False


    if not 'monitor_ff_layers' in options:
        options['monitor_ff_layers'] = False


