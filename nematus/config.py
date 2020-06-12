import argparse
import collections
import json
import logging
import pickle
import sys

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import util
except (ModuleNotFoundError, ImportError) as e:
    import util

class ParameterSpecification:
    """Describes a Nematus configuration parameter.

    For many parameters, a ParameterSpecification simply gets mapped to an
    argparse.add_argument() call when reading parameters from the command-line
    (as opposed to reading from a pre-existing config file). To make this
    convenient, ParameterSpecification's constructor accepts all of
    argparse.add_argument()'s keyword arguments so they can simply be passed
    through. For parameters with more complex defintions,
    ParameterSpecification adds some supporting arguments:

      - legacy_names: a ParameterSpecification can optionally include a list of
          legacy parameter names that will be used by
          load_config_from_json_file() to automatically recognise and update
          parameters with old names when reading from a JSON file.

      - visible_arg_names / hidden_arg_names: a ParameterSpecification can
          include multiple synonyms for the command-line argument.
          read_config_from_cmdline() will automatically add these to the
          parser, making them visible (via train.py -h, etc.) or hidden from
          users.

      - derivation_func: a few parameters are derived using the values of other
          parameters after the initial pass (i.e. after argparse has parsed the
          command-line arguments or after the parameters have been loaded from
          a pre-existing JSON config). For instance, if dim_per_factor is not
          set during the initial pass then it is set to [embedding_size]
          (provided that factors == 1).

    Note that unlike argparse.add_argument(), it is required to supply a
    default value. Generally, we need a default value both for
    argparse.add_argument() and also to fill in a missing parameter value when
    reading a config from an older JSON file.

    Some parameters don't have corresponding command-line arguments (e.g.
    model_version). They can be represented as ParameterSpecification objects
    by leaving both visible_arg_names and hidden_arg_names empty.
    """

    def __init__(self, name, default, legacy_names=[], visible_arg_names=[],
                 hidden_arg_names=[], derivation_func=None, **argparse_args):
        """
        Args:
            name: string (must be a valid Python variable name).
            default: the default parameter value.
            legacy_names: list of strings.
            visible_arg_names: list of strings (all must start '-' or '--')
            hidden_arg_names: list of strings (all must start '-' or '--')
            derivation_func: function taking config and meta_config arguments.
            argparse_args: any keyword arguments accepted by argparse.
        """
        self.name = name
        self.default = default
        self.legacy_names = legacy_names
        self.visible_arg_names = visible_arg_names
        self.hidden_arg_names = hidden_arg_names
        self.derivation_func = derivation_func
        self.argparse_args = argparse_args
        if len(argparse_args) == 0:
            assert visible_arg_names == [] and hidden_arg_names == []
        else:
            self.argparse_args['default'] = self.default


class ConfigSpecification:
    """A collection of ParameterSpecifications representing a complete config.

    The ParameterSpecifications are organised into groups. These are used with
    argparse's add_argument_group() mechanism when constructing a command-line
    argument parser (in read_config_from_cmdline()). They don't serve any
    other role.

    The nameless '' group is used for top-level command-line arguments (or it
    would be if we had any) and for parameters that don't have corresponding
    command-line arguments.
    """

    def __init__(self):
        """Builds the collection of ParameterSpecifications."""

        # Define the parameter groups and their descriptions.
        description_pairs = [
            ('',                      None),
            ('data',                  'data sets; model loading and saving'),
            ('network',               'network parameters (all model types)'),
            ('network_rnn',           'network parameters (rnn-specific)'),
            ('network_transformer',   'network parameters (transformer-'
                                      'specific)'),
            ('training',              'training parameters'),
            ('validation',            'validation parameters'),
            ('display',               'display parameters'),
            ('translate',             'translate parameters'),
            ('sampling',              'sampling parameters'),
            ('MRT',                   'MRT parameters'),
        ]
        self._group_descriptions = collections.OrderedDict(description_pairs)

        # Add all the ParameterSpecification objects.
        self._param_specs = self._define_param_specs()

        # Check that there are no duplicated names.
        self._check_self()

        # Build a dictionary for looking up ParameterSpecifications by name.
        self._name_to_spec = self._build_name_to_spec()

    @property
    def group_names(self):
        """Returns the list of parameter group names."""
        return self._group_descriptions.keys()

    def group_description(self, name):
        """Returns the description string for the given group name."""
        return self._group_descriptions[name]

    def params_by_group(self, group_name):
        """Returns the list of ParameterSpecifications for the given group."""
        return self._param_specs[group_name]

    def lookup(self, name):
        """Looks up a ParameterSpecification by name. None if not found."""
        return self._name_to_spec.get(name, None)

    def _define_param_specs(self):
        """Adds all ParameterSpecification objects."""
        param_specs = {}

        # Add an empty list for each parameter group.
        for group in self.group_names:
            param_specs[group] = []

        # Add non-command-line parameters.

        group = param_specs['']

        group.append(ParameterSpecification(
            name='model_version', default=None,
            derivation_func=_derive_model_version))

        group.append(ParameterSpecification(
            name='theano_compat', default=None,
            derivation_func=lambda _, meta_config: meta_config.from_theano))

        group.append(ParameterSpecification(
            name='source_dicts', default=None,
            derivation_func=lambda config, _: config.dictionaries[:-1]))

        group.append(ParameterSpecification(
            name='target_dict', default=None,
            derivation_func=lambda config, _: config.dictionaries[-1]))

        group.append(ParameterSpecification(
            name='target_embedding_size', default=None,
            derivation_func=_derive_target_embedding_size))

        # All remaining parameters are command-line parameters.

        # Add command-line parameters for the 'data' group.

        group = param_specs['data']

        group.append(ParameterSpecification(
            name='source_dataset', default=None,
            visible_arg_names=['--source_dataset'],
            derivation_func=_derive_source_dataset,
            type=str, metavar='PATH',
            help='parallel training corpus (source)'))

        group.append(ParameterSpecification(
            name='target_dataset', default=None,
            visible_arg_names=['--target_dataset'],
            derivation_func=_derive_target_dataset,
            type=str, metavar='PATH',
            help='parallel training corpus (target)'))

        # Hidden option for backward compatibility.
        group.append(ParameterSpecification(
            name='datasets', default=None,
            visible_arg_names=[], hidden_arg_names=['--datasets'],
            type=str, metavar='PATH', nargs=2))

        group.append(ParameterSpecification(
            name='dictionaries', default=None,
            visible_arg_names=['--dictionaries'], hidden_arg_names=[],
            type=str, required=True, metavar='PATH', nargs='+',
            help='network vocabularies (one per source factor, plus target '
                 'vocabulary)'))

        group.append(ParameterSpecification(
            name='save_freq', default=30000,
            legacy_names=['saveFreq'],
            visible_arg_names=['--save_freq'], hidden_arg_names=['--saveFreq'],
            type=int, metavar='INT',
            help='save frequency (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='saveto', default='model',
            visible_arg_names=['--model'], hidden_arg_names=['--saveto'],
            type=str, metavar='PATH',
            help='model file name (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='reload', default=None,
            visible_arg_names=['--reload'],
            type=str, metavar='PATH',
            help='load existing model from this path. Set to '
                 '"latest_checkpoint" to reload the latest checkpoint in the '
                 'same directory of --model'))

        group.append(ParameterSpecification(
            name='reload_training_progress', default=True,
            visible_arg_names=['--no_reload_training_progress'],
            action='store_false',
            help='don\'t reload training progress (only used if --reload '
                 'is enabled)'))

        group.append(ParameterSpecification(
            name='summary_dir', default=None,
            visible_arg_names=['--summary_dir'],
            type=str, metavar='PATH',
            help='directory for saving summaries (default: same directory '
                 'as the --model file)'))

        group.append(ParameterSpecification(
            name='summary_freq', default=0,
            legacy_names=['summaryFreq'],
            visible_arg_names=['--summary_freq'],
            hidden_arg_names=['--summaryFreq'],
            type=int, metavar='INT',
            help='Save summaries after INT updates, if 0 do not save '
                 'summaries (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='preprocess_script', default=None,
            visible_arg_names=['--preprocess_script'],
            type=str, metavar='PATH',
            help='path to script for external preprocessing (default: '
                 '%(default)s). The script will be called at the start of training, and before each epoch. '
                 'Useful for dynamic preprocessing, such as BPE dropout. Ideally, this script should write the files '
                 'given in --source_dataset and --target_dataset, which will be reloaded after calling the script.'))

        # Add command-line parameters for 'network' group.

        group = param_specs['network']

        group.append(ParameterSpecification(
            name='model_type', default='rnn',
            visible_arg_names=['--model_type'],
            type=str, choices=['rnn', 'transformer'],
            help='model type (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='embedding_size', default=512,
            legacy_names=['dim_word'],
            visible_arg_names=['--embedding_size'],
            hidden_arg_names=['--dim_word'],
            type=int, metavar='INT',
            help='embedding layer size (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='state_size', default=1000,
            legacy_names=['dim'],
            visible_arg_names=['--state_size'], hidden_arg_names=['--dim'],
            type=int, metavar='INT',
            help='hidden state size (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='source_vocab_sizes', default=None,
            visible_arg_names=['--source_vocab_sizes'],
            hidden_arg_names=['--n_words_src'],
            derivation_func=_derive_source_vocab_sizes,
            type=int, metavar='INT', nargs='+',
            help='source vocabulary sizes (one per input factor) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='target_vocab_size', default=-1,
            legacy_names=['n_words'],
            visible_arg_names=['--target_vocab_size'],
            hidden_arg_names=['--n_words'],
            derivation_func=_derive_target_vocab_size,
            type=int, metavar='INT',
            help='target vocabulary size (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='factors', default=1,
            visible_arg_names=['--factors'],
            type=int, metavar='INT',
            help='number of input factors (default: %(default)s) - CURRENTLY '
                 'ONLY WORKS FOR \'rnn\' MODEL'))

        group.append(ParameterSpecification(
            name='dim_per_factor', default=None,
            visible_arg_names=['--dim_per_factor'],
            derivation_func=_derive_dim_per_factor,
            type=int, metavar='INT', nargs='+',
            help='list of word vector dimensionalities (one per factor): '
                 '\'--dim_per_factor 250 200 50\' for total dimensionality '
                 'of 500 (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='tie_encoder_decoder_embeddings', default=False,
            visible_arg_names=['--tie_encoder_decoder_embeddings'],
            action='store_true',
            help='tie the input embeddings of the encoder and the decoder '
                 '(first factor only). Source and target vocabulary size '
                 'must be the same'))

        group.append(ParameterSpecification(
            name='tie_decoder_embeddings', default=False,
            visible_arg_names=['--tie_decoder_embeddings'],
            action='store_true',
            help='tie the input embeddings of the decoder with the softmax '
                 'output embeddings'))

        group.append(ParameterSpecification(
            name='output_hidden_activation', default='tanh',
            visible_arg_names=['--output_hidden_activation'],
            type=str, choices=['tanh', 'relu', 'prelu', 'linear'],
            help='activation function in hidden layer of the output '
                 'network (default: %(default)s) - CURRENTLY ONLY WORKS '
                 'FOR \'rnn\' MODEL'))

        group.append(ParameterSpecification(
            name='softmax_mixture_size', default=1,
            visible_arg_names=['--softmax_mixture_size'],
            type=int, metavar='INT',
            help='number of softmax components to use (default: '
                 '%(default)s) - CURRENTLY ONLY WORKS FOR \'rnn\' MODEL'))

        # Add command-line parameters for 'network_rnn' group.

        group = param_specs['network_rnn']

        # NOTE: parameter names in this group must use the rnn_ prefix.
        #       read_config_from_cmdline() uses this to check that only
        #       model type specific options are only used with the appropriate
        #       model type.

        group.append(ParameterSpecification(
            name='rnn_enc_depth', default=1,
            legacy_names=['enc_depth'],
            visible_arg_names=['--rnn_enc_depth'],
            hidden_arg_names=['--enc_depth'],
            type=int, metavar='INT',
            help='number of encoder layers (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_enc_transition_depth', default=1,
            legacy_names=['enc_recurrence_transition_depth'],
            visible_arg_names=['--rnn_enc_transition_depth'],
            hidden_arg_names=['--enc_recurrence_transition_depth'],
            type=int, metavar='INT',
            help='number of GRU transition operations applied in the '
                 'encoder. Minimum is 1. (Only applies to gru). (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dec_depth', default=1,
            legacy_names=['dec_depth'],
            visible_arg_names=['--rnn_dec_depth'],
            hidden_arg_names=['--dec_depth'],
            type=int, metavar='INT',
            help='number of decoder layers (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dec_base_transition_depth', default=2,
            legacy_names=['dec_base_recurrence_transition_depth'],
            visible_arg_names=['--rnn_dec_base_transition_depth'],
            hidden_arg_names=['--dec_base_recurrence_transition_depth'],
            type=int, metavar='INT',
            help='number of GRU transition operations applied in the first '
                 'layer of the decoder. Minimum is 2.  (Only applies to '
                 'gru_cond). (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dec_high_transition_depth', default=1,
            legacy_names=['dec_high_recurrence_transition_depth'],
            visible_arg_names=['--rnn_dec_high_transition_depth'],
            hidden_arg_names=['--dec_high_recurrence_transition_depth'],
            type=int, metavar='INT',
            help='number of GRU transition operations applied in the higher '
                 'layers of the decoder. Minimum is 1. (Only applies to '
                 'gru). (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dec_deep_context', default=False,
            legacy_names=['dec_deep_context'],
            visible_arg_names=['--rnn_dec_deep_context'],
            hidden_arg_names=['--dec_deep_context'],
            action='store_true',
            help='pass context vector (from first layer) to deep decoder '
                 'layers'))

        # option should no longer be set in command line;
        # code only remains to ensure backward-compatible loading of JSON files
        group.append(ParameterSpecification(
            name='rnn_use_dropout', default=False,
            legacy_names=['use_dropout'],
            visible_arg_names=['--rnn_use_dropout'],
            hidden_arg_names=['--use_dropout'],
            action='store_true',
            help='REMOVED: has no effect'))

        group.append(ParameterSpecification(
            name='rnn_dropout_embedding', default=0.0,
            legacy_names=['dropout_embedding'],
            visible_arg_names=['--rnn_dropout_embedding'],
            hidden_arg_names=['--dropout_embedding'],
            type=float, metavar='FLOAT',
            help='dropout for input embeddings (0: no dropout) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dropout_hidden', default=0.0,
            legacy_names=['dropout_hidden'],
            visible_arg_names=['--rnn_dropout_hidden'],
            hidden_arg_names=['--dropout_hidden'],
            type=float, metavar='FLOAT',
            help='dropout for hidden layer (0: no dropout) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dropout_source', default=0.0,
            legacy_names=['dropout_source'],
            visible_arg_names=['--rnn_dropout_source'],
            hidden_arg_names=['--dropout_source'],
            type=float, metavar='FLOAT',
            help='dropout source words (0: no dropout) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_dropout_target', default=0.0,
            legacy_names=['dropout_target'],
            visible_arg_names=['--rnn_dropout_target'],
            hidden_arg_names=['--dropout_target'],
            type=float, metavar='FLOAT',
            help='dropout target words (0: no dropout) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='rnn_layer_normalization', default=False,
            legacy_names=['use_layer_norm', 'layer_normalisation'],
            visible_arg_names=['--rnn_layer_normalisation'],
            hidden_arg_names=['--use_layer_norm', '--layer_normalisation'],
            action='store_true',
            help='Set to use layer normalization in encoder and decoder'))

        group.append(ParameterSpecification(
            name='rnn_lexical_model', default=False,
            legacy_names=['lexical_model'],
            visible_arg_names=['--rnn_lexical_model'],
            hidden_arg_names=['--lexical_model'],
            action='store_true',
            help='Enable feedforward lexical model (Nguyen and Chiang, 2018)'))

        # Add command-line parameters for 'network_transformer' group.

        group = param_specs['network_transformer']

        # NOTE: parameter names in this group must use the transformer_ prefix.
        #       read_config_from_cmdline() uses this to check that only
        #       model type specific options are only used with the appropriate
        #       model type.

        group.append(ParameterSpecification(
            name='transformer_enc_depth', default=6,
            visible_arg_names=['--transformer_enc_depth'],
            type=int, metavar='INT',
            help='number of encoder layers (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_dec_depth', default=6,
            visible_arg_names=['--transformer_dec_depth'],
            type=int, metavar='INT',
            help='number of decoder layers (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_ffn_hidden_size', default=2048,
            visible_arg_names=['--transformer_ffn_hidden_size'],
            type=int, metavar='INT',
            help='inner dimensionality of feed-forward sub-layers (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_num_heads', default=8,
            visible_arg_names=['--transformer_num_heads'],
            type=int, metavar='INT',
            help='number of attention heads used in multi-head attention '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_dropout_embeddings', default=0.1,
            visible_arg_names=['--transformer_dropout_embeddings'],
            type=float, metavar='FLOAT',
            help='dropout applied to sums of word embeddings and positional '
                 'encodings (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_dropout_residual', default=0.1,
            visible_arg_names=['--transformer_dropout_residual'],
            type=float, metavar='FLOAT',
            help='dropout applied to residual connections (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_dropout_relu', default=0.1,
            visible_arg_names=['--transformer_dropout_relu'],
            type=float, metavar='FLOAT',
            help='dropout applied to the internal activation of the '
                 'feed-forward sub-layers (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_dropout_attn', default=0.1,
            visible_arg_names=['--transformer_dropout_attn'],
            type=float, metavar='FLOAT',
            help='dropout applied to attention weights (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='transformer_drophead', default=0.0,
            visible_arg_names=['--transformer_drophead'],
            type=float, metavar='FLOAT',
            help='dropout of entire attention heads (default: '
                 '%(default)s)'))

        # Add command-line parameters for 'training' group.

        group = param_specs['training']

        group.append(ParameterSpecification(
            name='loss_function', default='cross-entropy',
            visible_arg_names=['--loss_function'],
            type=str, choices=['cross-entropy', 'per-token-cross-entropy', 'MRT'],
            help='loss function (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='decay_c', default=0.0,
            visible_arg_names=['--decay_c'],
            type=float, metavar='FLOAT',
            help='L2 regularization penalty (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='map_decay_c', default=0.0,
            visible_arg_names=['--map_decay_c'],
            type=float, metavar='FLOAT',
            help='MAP-L2 regularization penalty towards original weights '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='prior_model', default=None,
            visible_arg_names=['--prior_model'],
            type=str, metavar='PATH',
            help='Prior model for MAP-L2 regularization. Unless using '
                 '\"--reload\", this will also be used for initialization.'))

        group.append(ParameterSpecification(
            name='clip_c', default=1.0,
            visible_arg_names=['--clip_c'],
            type=float, metavar='FLOAT',
            help='gradient clipping threshold (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='label_smoothing', default=0.0,
            visible_arg_names=['--label_smoothing'],
            type=float, metavar='FLOAT',
            help='label smoothing (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='exponential_smoothing', default=0.0,
            visible_arg_names=['--exponential_smoothing'],
            type=float, metavar='FLOAT',
            help='exponential smoothing factor; use 0 to disable (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='optimizer', default='adam',
            visible_arg_names=['--optimizer'],
            type=str, choices=['adam'],
            help='optimizer (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='adam_beta1', default=0.9,
            visible_arg_names=['--adam_beta1'],
            type=float, metavar='FLOAT',
            help='exponential decay rate for the first moment estimates '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='adam_beta2', default=0.999,
            visible_arg_names=['--adam_beta2'],
            type=float, metavar='FLOAT',
            help='exponential decay rate for the second moment estimates '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='adam_epsilon', default=1e-08,
            visible_arg_names=['--adam_epsilon'],
            type=float, metavar='FLOAT',
            help='constant for numerical stability (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='learning_schedule', default='constant',
            visible_arg_names=['--learning_schedule'],
            type=str, choices=['constant', 'transformer',
                               'warmup-plateau-decay'],
            help='learning schedule (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='learning_rate', default=0.0001,
            visible_arg_names=['--learning_rate'],
            hidden_arg_names=['--lrate'],
            legacy_names=['lrate'],
            type=float, metavar='FLOAT',
            help='learning rate (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='warmup_steps', default=8000,
            visible_arg_names=['--warmup_steps'],
            type=int, metavar='INT',
            help='number of initial updates during which the learning rate is '
                 'increased linearly during learning rate scheduling '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='plateau_steps', default=0,
            visible_arg_names=['--plateau_steps'],
            type=int, metavar='INT',
            help='number of updates after warm-up before the learning rate '
                 'starts to decay (applies to \'warmup-plateau-decay\' '
                 'learning schedule only). (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='maxlen', default=100,
            visible_arg_names=['--maxlen'],
            type=int, metavar='INT',
            help='maximum sequence length for training and validation '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='batch_size', default=80,
            visible_arg_names=['--batch_size'],
            type=int, metavar='INT',
            help='minibatch size (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='token_batch_size', default=0,
            visible_arg_names=['--token_batch_size'],
            type=int, metavar='INT',
            help='minibatch size (expressed in number of source or target '
                 'tokens). Sentence-level minibatch size will be dynamic. If '
                 'this is enabled, batch_size only affects sorting by '
                 'length. (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='max_sentences_per_device', default=0,
            visible_arg_names=['--max_sentences_per_device'],
            type=int, metavar='INT',
            help='maximum size of minibatch subset to run on a single device, '
                 'in number of sentences (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='max_tokens_per_device', default=0,
            visible_arg_names=['--max_tokens_per_device'],
            type=int, metavar='INT',
            help='maximum size of minibatch subset to run on a single device, '
                 'in number of tokens (either source or target - whichever is '
                 'highest) (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='gradient_aggregation_steps', default=1,
            visible_arg_names=['--gradient_aggregation_steps'],
            type=int, metavar='INT',
            help='number of times to accumulate gradients before aggregating '
                 'and applying; the minibatch is split between steps, so '
                 'adding more steps allows larger minibatches to be used '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='maxibatch_size', default=20,
            visible_arg_names=['--maxibatch_size'],
            type=int, metavar='INT',
            help='size of maxibatch (number of minibatches that are sorted '
                 'by length) (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='sort_by_length', default=True,
            visible_arg_names=['--no_sort_by_length'],
            action='store_false',
            help='do not sort sentences in maxibatch by length'))

        group.append(ParameterSpecification(
            name='shuffle_each_epoch', default=True,
            visible_arg_names=['--no_shuffle'],
            action='store_false',
            help='disable shuffling of training data (for each epoch)'))

        group.append(ParameterSpecification(
            name='keep_train_set_in_memory', default=False,
            visible_arg_names=['--keep_train_set_in_memory'],
            action='store_true',
            help='Keep training dataset lines stores in RAM during training'))

        group.append(ParameterSpecification(
            name='max_epochs', default=5000,
            visible_arg_names=['--max_epochs'],
            type=int, metavar='INT',
            help='maximum number of epochs (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='finish_after', default=10000000,
            visible_arg_names=['--finish_after'],
            type=int, metavar='INT',
            help='maximum number of updates (minibatches) (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='print_per_token_pro', default=False,
            visible_arg_names=['--print_per_token_pro'],
            type=str,
            help='PATH to store the probability of each target token given source sentences '
                 'over the training dataset (default: %(default)s). Please get rid of the 1.0s at the end '
                 'of each list which is the probability of padding.'))

        # Add command-line parameters for 'validation' group.

        group = param_specs['validation']

        group.append(ParameterSpecification(
            name='valid_source_dataset', default=None,
            visible_arg_names=['--valid_source_dataset'],
            derivation_func=_derive_valid_source_dataset,
            type=str, metavar='PATH',
            help='source validation corpus (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='valid_bleu_source_dataset', default=None,
            visible_arg_names=['--valid_bleu_source_dataset'],
            derivation_func=_derive_valid_source_bleu_dataset,
            type=str, metavar='PATH',
            help='source validation corpus for external evaluation bleu (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='valid_target_dataset', default=None,
            visible_arg_names=['--valid_target_dataset'],
            derivation_func=_derive_valid_target_dataset,
            type=str, metavar='PATH',
            help='target validation corpus (default: %(default)s)'))

        # Hidden option for backward compatibility.
        group.append(ParameterSpecification(
            name='valid_datasets', default=None,
            hidden_arg_names=['--valid_datasets'],
            type=str, metavar='PATH', nargs=2))

        group.append(ParameterSpecification(
            name='valid_batch_size', default=80,
            visible_arg_names=['--valid_batch_size'],
            type=int, metavar='INT',
            help='validation minibatch size (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='valid_token_batch_size', default=0,
            visible_arg_names=['--valid_token_batch_size'],
            type=int, metavar='INT',
            help='validation minibatch size (expressed in number of source '
                 'or target tokens). Sentence-level minibatch size will be '
                 'dynamic. If this is enabled, valid_batch_size only affects '
                 'sorting by length. (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='valid_freq', default=10000,
            legacy_names=['validFreq'],
            visible_arg_names=['--valid_freq'],
            hidden_arg_names=['--validFreq'],
            type=int, metavar='INT',
            help='validation frequency (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='valid_script', default=None,
            visible_arg_names=['--valid_script'],
            type=str, metavar='PATH',
            help='path to script for external validation (default: '
                 '%(default)s). The script will be passed an argument '
                 'specifying the path of a file that contains translations '
                 'of the source validation corpus. It must write a single '
                 'score to standard output.'))

        group.append(ParameterSpecification(
            name='patience', default=10,
            visible_arg_names=['--patience'],
            type=int, metavar='INT',
            help='early stopping patience (default: %(default)s)'))

        # Add command-line parameters for 'display' group.

        group = param_specs['display']

        group.append(ParameterSpecification(
            name='disp_freq', default=1000,
            legacy_names=['dispFreq'],
            visible_arg_names=['--disp_freq'], hidden_arg_names=['--dispFreq'],
            type=int, metavar='INT',
            help='display loss after INT updates (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='sample_freq', default=10000,
            legacy_names=['sampleFreq'],
            visible_arg_names=['--sample_freq'],
            hidden_arg_names=['--sampleFreq'],
            type=int, metavar='INT',
            help='display some samples after INT updates (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='beam_freq', default=10000,
            legacy_names=['beamFreq'],
            visible_arg_names=['--beam_freq'], hidden_arg_names=['--beamFreq'],
            type=int, metavar='INT',
            help='display some beam_search samples after INT updates '
                 '(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='beam_size', default=12,
            visible_arg_names=['--beam_size'],
            type=int, metavar='INT',
            help='size of the beam (default: %(default)s)'))

        # Add command-line parameters for 'translate' group.

        group = param_specs['translate']

        group.append(ParameterSpecification(
            name='normalization_alpha', type=float, default=0.0, nargs="?",
            const=1.0, metavar="ALPHA",
            visible_arg_names=['--normalization_alpha'],
            help='normalize scores by sentence length (with argument, " \
                 "exponentiate lengths by ALPHA)'))

        group.append(ParameterSpecification(
            name='n_best', default=False,
            visible_arg_names=['--n_best'],
            action='store_true', dest='n_best',
            help='Print full beam'))

        group.append(ParameterSpecification(
            name='translation_maxlen', default=200,
            visible_arg_names=['--translation_maxlen'],
            type=int, metavar='INT',
            help='Maximum length of translation output sentence (default: '
                 '%(default)s)'))

        group.append(ParameterSpecification(
            name='translation_strategy', default='beam_search',
            visible_arg_names=['--translation_strategy'],
            type=str, choices=['beam_search', 'sampling'],
            help='translation_strategy, either beam_search or sampling (default: %(default)s)'))

        # Add Add command-line parameters for 'MRT' group.

        group = param_specs['MRT']

        group.append(ParameterSpecification(
            name='mrt_reference', default=False,
            visible_arg_names=['--mrt_reference'],
            action='store_true',
            help='add reference into MRT candidates sentences'))

        group.append(ParameterSpecification(
            name='mrt_alpha', default=0.005,
            visible_arg_names=['--mrt_alpha'],
            type=float, metavar='FLOAT',
            help='MRT alpha to control sharpness ofthe distribution of '
                 'sampled subspace(default: %(default)s)'))

        group.append(ParameterSpecification(
            name='samplesN', default=100,
            visible_arg_names=['--samplesN'],
            type=int, metavar='INT',
            help='the number of sampled candidates sentences per source sentence (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='mrt_loss', default='SENTENCEBLEU n=4',
            visible_arg_names=['--mrt_loss'],
            type=str, metavar='STR',
            help='evaluation matrics used in MRT (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='mrt_ml_mix', default=0,
            visible_arg_names=['--mrt_ml_mix'],
            type=float, metavar='FLOAT',
            help='mix in MLE objective in MRT training with this scaling factor (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='sample_way', default='beam_search',
            visible_arg_names=['--sample_way'],
            type=str, choices=['beam_search', 'randomly_sample'],
            help='the sampling strategy to generate candidates sentences (default: %(default)s)'))

        group.append(ParameterSpecification(
            name='max_len_a', default=1.5,
            visible_arg_names=['--max_len_a'],
            type=float, metavar='FLOAT',
            help='generate candidates sentences with maximum length: ax + b, '
                             'where x is the source length'))

        group.append(ParameterSpecification(
            name='max_len_b', default=5,
            visible_arg_names=['--max_len_b'],
            type=int, metavar='INT',
            help='generate candidates sentences with maximum length ax + b, '
                             'where x is the source length'))

        group.append(ParameterSpecification(
            name='max_sentences_of_sampling', default=0,
            visible_arg_names=['--max_sentences_of_sampling'],
            type=int, metavar='INT',
            help='maximum number of source sentences to generate candidates sentences '
                 'at one time (limited by device memory capacity) (default: %(default)s)'))

        # Add command-line parameters for 'sampling' group.

        group = param_specs['sampling']

        group.append(ParameterSpecification(
            name='sampling_temperature', type=float, default=1.0,
            metavar="FLOAT",
            visible_arg_names=['--sampling_temperature'],
            help='softmax temperature used for sampling (default %(default)s)'))

        return param_specs

    def _build_name_to_spec(self):
        name_to_spec = {}
        for group in self.group_names:
            for param in self.params_by_group(group):
                for name in [param.name] + param.legacy_names:
                    assert name not in name_to_spec
                    name_to_spec[name] = param
        return name_to_spec

    def _check_self(self):
        # Check that there are no duplicated parameter names.
        param_names = set()
        for group in self.group_names:
            for param in self.params_by_group(group):
                assert param.name not in param_names
                param_names.add(param.name)
                for name in param.legacy_names:
                    assert name not in param_names
                    param_names.add(name)
        # Check that there are no duplicated command-line argument names.
        arg_names = set()
        for group in self.group_names:
            for param in self.params_by_group(group):
                for arg_list in (param.visible_arg_names,
                                 param.hidden_arg_names):
                    for name in arg_list:
                        assert name not in arg_names
                        arg_names.add(param.name)


def _construct_argument_parser(spec, suppress_missing=False):
    """Constructs an argparse.ArgumentParser given a ConfigSpecification.

    Setting suppress_missing to True causes the parser to suppress arguments
    that are not supplied by the user (as opposed to adding them with
    their default values).

    Args:
        spec: a ConfigSpecification object.
        suppress_missing: Boolean

    Returns:
        An argparse.ArgumentParser.
    """
    # Construct an ArgumentParser and parse command-line args.
    parser = argparse.ArgumentParser()
    for group_name in spec.group_names:
        if group_name == "":
            target = parser
        else:
            description = spec.group_description(group_name)
            target = parser.add_argument_group(description)

        for param in spec.params_by_group(group_name):
            if param.visible_arg_names == [] and param.hidden_arg_names == []:
                # Internal parameter - no command-line argument.
                continue
            argparse_args = dict(param.argparse_args)
            argparse_args['dest'] = param.name
            if suppress_missing:
                argparse_args['default'] = argparse.SUPPRESS
            if param.visible_arg_names == []:
                argparse_args['help'] = argparse.SUPPRESS
                target.add_argument(*param.hidden_arg_names, **argparse_args)
                continue
            if 'required' in argparse_args and argparse_args['required']:
                mutex_group = \
                    target.add_mutually_exclusive_group(required=True)
                del argparse_args['required']
            else:
                mutex_group = target.add_mutually_exclusive_group()
            mutex_group.add_argument(*param.visible_arg_names, **argparse_args)
            # Add any hidden arguments for this param.
            if len(param.hidden_arg_names) > 0:
                argparse_args['help'] = argparse.SUPPRESS
                mutex_group.add_argument(*param.hidden_arg_names,
                                         **argparse_args)
    return parser


def read_config_from_cmdline():
    """Reads a config from the command-line.

    Logs an error and exits if the parameter values are not mutually
    consistent.

    Returns:
        An argparse.Namespace object representing the config.
    """

    spec = ConfigSpecification()

    # Construct an argparse.ArgumentParser and parse command-line args.
    parser = _construct_argument_parser(spec)
    config = parser.parse_args()

    # Construct a second ArgumentParser but using default=argparse.SUPPRESS
    # in every argparse.add_argument() call. This allows us to determine
    # which parameters were actually set by the user.
    # Solution is from https://stackoverflow.com/a/45803037
    aux_parser = _construct_argument_parser(spec, suppress_missing=True)
    aux_config = aux_parser.parse_args()
    set_by_user = set(vars(aux_config).keys())

    # Perform consistency checks.
    error_messages = _check_config_consistency(spec, config, set_by_user)
    if len(error_messages) > 0:
        for msg in error_messages:
            logging.error(msg)
        sys.exit(1)

    # Set meta parameters.
    meta_config = argparse.Namespace()
    meta_config.from_cmdline = True
    meta_config.from_theano = False

    # Set defaults for removed options
    config.rnn_use_dropout = True

    # Run derivation functions.
    for group in spec.group_names:
        for param in spec.params_by_group(group):
            if param.derivation_func is not None:
                setattr(config, param.name,
                        param.derivation_func(config, meta_config))

    return config


def write_config_to_json_file(config, path):
    """
    Writes a config object to a JSON file.

    Args:
        config: a config Namespace object
        path: full path to the JSON file except ".json" suffix
    """

    config_as_dict = collections.OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('%s.json' % path, 'w', encoding="UTF-8"), indent=2)


def load_config_from_json_file(basename):
    """Loads and, if necessary, updates a config from a JSON (or Pickle) file.

    Logs an error and exits if the file can't be loaded.

    Args:
        basename: a string containing the path to the corresponding model file.

    Returns:
        An argparse.Namespace object representing the config.
    """

    spec = ConfigSpecification()

    # Load a config from a JSON (or Pickle) config file.
    try:
        with open('%s.json' % basename, 'r', encoding='utf-8') as f:
            config_as_dict = json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'r', encoding='utf-8') as f:
                config_as_dict = pickle.load(f)
        except:
            logging.error('config file {}.json is missing'.format(basename))
            sys.exit(1)
    config = argparse.Namespace(**config_as_dict)

    # Set meta parameters.
    meta_config = argparse.Namespace()
    meta_config.from_cmdline = False
    meta_config.from_theano = (not hasattr(config, 'embedding_size'))

    # Update config to use current parameter names.
    for group_name in spec.group_names:
        for param in spec.params_by_group(group_name):
            for legacy_name in param.legacy_names:
                # TODO It shouldn't happen, but check for multiple names
                #      (legacy and/or current) for same parameter appearing
                #      in config.
                if hasattr(config, legacy_name):
                    val = getattr(config, legacy_name)
                    assert not hasattr(config, param.name)
                    setattr(config, param.name, val)
                    delattr(config, legacy_name)

    # Add missing parameters.
    for group_name in spec.group_names:
        for param in spec.params_by_group(group_name):
            if not hasattr(config, param.name):
                setattr(config, param.name, param.default)

    # Run derivation functions.
    for group in spec.group_names:
        for param in spec.params_by_group(group):
            if param.derivation_func is not None:
                setattr(config, param.name,
                        param.derivation_func(config, meta_config))

    return config


def _check_config_consistency(spec, config, set_by_user):
    """Performs consistency checks on a config read from the command-line.

    Args:
        spec: a ConfigSpecification object.
        config: an argparse.Namespace object.
        set_by_user: a set of strings representing parameter names.

    Returns:
        A list of error messages, one for each check that failed. An empty
        list indicates that all checks passed.
    """

    def arg_names_string(param):
        arg_names = param.visible_arg_names + param.hidden_arg_names
        return ' / '.join(arg_names)

    error_messages = []

    # Check parameters are appropriate for the model type.
    assert config.model_type is not None
    for group in spec.group_names:
        for param in spec.params_by_group(group):
            if param.name not in set_by_user:
                continue
            if ((param.name.startswith('rnn_') and
                 config.model_type == 'transformer') or
                 (param.name.startswith('transformer_') and
                 config.model_type == 'rnn')):
                msg = '{} cannot be used with \'{}\' model type'.format(
                    arg_names_string(param), config.model_type)
                error_messages.append(msg)

    # Check user-supplied learning schedule options are consistent.
    if config.learning_schedule == 'constant':
        for key in ['warmup_steps', 'plateau_steps']:
            param = spec.lookup(key)
            assert param is not None
            if param.name in set_by_user:
                msg = '{} cannot be used with \'constant\' learning ' \
                       'schedule'.format(arg_names_string(param),
                                         config.model_type)
                error_messages.append(msg)
    elif config.learning_schedule == 'transformer':
        for key in ['learning_rate', 'plateau_steps']:
            param = spec.lookup(key)
            assert param is not None
            if param.name in set_by_user:
                msg = '{} cannot be used with \'transformer\' learning ' \
                      'schedule'.format(arg_names_string(param),
                                        config.model_type)
                error_messages.append(msg)

    # TODO Other similar checks? e.g. check user hasn't set adam parameters
    #       if optimizer != 'adam' (not currently possible but probably will
    #       be in in the future)...

    # Check if user is trying to use the Transformer with features that
    # aren't supported yet.
    if config.model_type == 'transformer':
        if config.factors > 1:
            msg = 'factors are not yet supported for the \'transformer\' ' \
                  'model type'
            error_messages.append(msg)
        if config.softmax_mixture_size > 1:
            msg = 'softmax mixtures are not yet supported for the ' \
                  '\'transformer\' model type'
            error_messages.append(msg)

    if config.datasets:
        if config.source_dataset or config.target_dataset:
            msg = 'argument clash: --datasets is mutually exclusive ' \
                  'with --source_dataset and --target_dataset'
            error_messages.append(msg)
    elif not config.source_dataset:
        msg = '--source_dataset is required'
        error_messages.append(msg)
    elif not config.target_dataset:
        msg = '--target_dataset is required'
        error_messages.append(msg)

    if config.valid_datasets:
        if config.valid_source_dataset or config.valid_target_dataset:
            msg = 'argument clash: --valid_datasets is mutually ' \
                  'exclusive with --valid_source_dataset and ' \
                  '--valid_target_dataset'
            error_messages.append(msg)

    if (config.source_vocab_sizes is not None and
            len(config.source_vocab_sizes) > config.factors):
        msg = 'too many values supplied to \'--source_vocab_sizes\' option ' \
              '(expected one per factor = {})'.format(config.factors)
        error_messages.append(msg)

    if config.dim_per_factor is None and config.factors != 1:
        msg = 'if using factored input, you must specify \'dim_per_factor\''
        error_messages.append(msg)

    if config.dim_per_factor is not None:
        if len(config.dim_per_factor) != config.factors:
            msg = 'mismatch between \'--factors\' ({0}) and ' \
                  '\'--dim_per_factor\' ({1} entries)'.format(
                      config.factors, len(config.dim_per_factor))
            error_messages.append(msg)
        elif sum(config.dim_per_factor) != config.embedding_size:
            msg = 'mismatch between \'--embedding_size\' ({0}) and ' \
                  '\'--dim_per_factor\' (sums to {1})\''.format(
                      config.embedding_size, sum(config.dim_per_factor))
            error_messages.append(msg)

    if len(config.dictionaries) != config.factors + 1:
        msg = '\'--dictionaries\' must specify one dictionary per source ' \
              'factor and one target dictionary'
        error_messages.append(msg)

    max_sents_param = spec.lookup('max_sentences_per_device')
    max_tokens_param = spec.lookup('max_tokens_per_device')

    # TODO Extend ParameterSpecification to support mutually exclusive
    #      command-line args.
    if (max_sents_param.name in set_by_user
        and max_tokens_param.name in set_by_user):
        msg = '{} is mutually exclusive with {}'.format(
            arg_names_string(max_sents_param),
            arg_names_string(max_tokens_param))
        error_messages.append(msg)

    aggregation_param = spec.lookup('gradient_aggregation_steps')

    if (aggregation_param.name in set_by_user
        and (max_sents_param.name in set_by_user
             or max_tokens_param.name in set_by_user)):
        msg = '{} is mutually exclusive with {} / {}'.format(
            arg_names_string(aggregation_param),
            arg_names_string(max_sents_param),
            arg_names_string(max_tokens_param))
        error_messages.append(msg)

    # softmax_mixture_size and lexical_model are currently mutually exclusive:
    if config.softmax_mixture_size > 1 and config.rnn_lexical_model:
       error_messages.append('behavior of --rnn_lexical_model is undefined if softmax_mixture_size > 1')

    if 'rnn_use_dropout' in set_by_user:
        msg = '--rnn_use_dropout is no longer used. Set --rnn_dropout_* instead (0 by default).\n' \
              'old defaults:\n' \
              '--rnn_dropout_embedding: 0.2\n' \
              '--rnn_dropout_hidden: 0.2\n' \
              '--rnn_dropout_source: 0\n' \
              '--rnn_dropout_target: 0'
        error_messages.append(msg)

    return error_messages


def _derive_model_version(config, meta_config):
    if meta_config.from_cmdline:
        # We're creating a new model - set the current version number.
        return 0.2
    if config.model_version is not None:
        return config.model_version
    if meta_config.from_theano and config.rnn_use_dropout:
        logging.error('version 0 dropout is not supported in '
                      'TensorFlow Nematus')
        sys.exit(1)
    return 0.1


def _derive_target_embedding_size(config, meta_config):
    assert hasattr(config, 'embedding_size')
    if not config.tie_encoder_decoder_embeddings:
        return config.embedding_size
    if config.factors > 1:
        assert hasattr(config, 'dim_per_factor')
        assert config.dim_per_factor is not None
        return config.dim_per_factor[0]
    else:
        return config.embedding_size


def _derive_source_dataset(config, meta_config):
    if config.source_dataset is not None:
        return config.source_dataset
    assert config.datasets is not None
    return config.datasets[0]


def _derive_target_dataset(config, meta_config):
    if config.target_dataset is not None:
        return config.target_dataset
    assert config.datasets is not None
    return config.datasets[1]


def _derive_source_vocab_sizes(config, meta_config):
    if config.source_vocab_sizes is not None:
        if len(config.source_vocab_sizes) == config.factors:
            # Case 1: we're loading parameters from a recent config or
            #         we're processing command-line arguments and
            #         a source_vocab_sizes was fully specified.
            return config.source_vocab_sizes
        else:
            # Case 2: source_vocab_sizes was given on the command-line
            #         but was only partially specified
            assert meta_config.from_cmdline
            assert len(config.source_vocab_sizes) < config.factors
            num_missing = config.factors - len(config.source_vocab_sizes)
            vocab_sizes = config.source_vocab_sizes + [-1] * num_missing
    elif hasattr(config, 'n_words_src'):
        # Case 3: we're loading parameters from a Theano config.
        #         This will always contain a single value for the
        #         source vocab size regardless of how many factors
        #         there are.
        assert not meta_config.from_cmdline
        assert meta_config.from_theano
        assert type(config.n_words_src) == int
        return [config.n_words_src] * config.factors
    elif hasattr(config, 'source_vocab_size'):
        # Case 4: we're loading parameters from a pre-factors
        #         TensorFlow config.
        assert not meta_config.from_cmdline
        assert not meta_config.from_theano
        assert config.factors == 1
        return [config.source_vocab_size]
    else:
        # Case 5: we're reading command-line parameters and
        #         --source_vocab_size was not given.
        assert meta_config.from_cmdline
        vocab_sizes = [-1] * config.factors
    # For any unspecified vocabulary sizes, determine sizes from the
    # vocabulary dictionaries.
    for i, vocab_size in enumerate(vocab_sizes):
        if vocab_size >= 0:
            continue
        path = config.dictionaries[i]
        vocab_sizes[i] = _determine_vocab_size_from_file(path,
                                                         config.model_type)
    return vocab_sizes


def _derive_target_vocab_size(config, meta_config):
    if config.target_vocab_size != -1:
        return config.target_vocab_size
    path = config.dictionaries[-1]
    return _determine_vocab_size_from_file(path, config.model_type)


def _derive_dim_per_factor(config, meta_config):
    if config.dim_per_factor is not None:
        return config.dim_per_factor
    assert config.factors == 1
    return [config.embedding_size]


def _derive_valid_source_dataset(config, meta_config):
    if config.valid_source_dataset is not None:
        return config.valid_source_dataset
    if config.valid_datasets is not None:
        return config.valid_datasets[0]
    return None

# if 'valid_bleu_source_dataset' is not declared, then set it same
# as 'valid_source_dataset'
def _derive_valid_source_bleu_dataset(config, meta_config):
    if config.valid_bleu_source_dataset is not None:
        return config.valid_bleu_source_dataset
    else:
        return config.valid_source_dataset


def _derive_valid_target_dataset(config, meta_config):
    if config.valid_target_dataset is not None:
        return config.valid_target_dataset
    if config.valid_datasets is not None:
        return config.valid_datasets[1]
    return None


def _determine_vocab_size_from_file(path, model_type):
    try:
        d = util.load_dict(path, model_type)
    except IOError as x:
        logging.error('failed to determine vocabulary size from file: '
                      '{}: {}'.format(path, str(x)))
        sys.exit(1)
    except:
        logging.error('failed to determine vocabulary size from file: '
                      '{}'.format(path))
        sys.exit(1)
    return max(d.values()) + 1
