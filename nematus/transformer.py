"""Adapted from Nematode: https://github.com/demelin/nematode """

import sys
import tensorflow as tf
import numpy

# ModuleNotFoundError is new in 3.6; older versions will throw SystemError
if sys.version_info < (3, 6):
    ModuleNotFoundError = SystemError

try:
    from . import model_inputs
    from . import mrt_utils as mru
    from .sampling_utils import SamplingUtils
    from . import tf_utils
    from .transformer_blocks import AttentionBlock, FFNBlock
    from .transformer_layers import \
        EmbeddingLayer, \
        MaskedCrossEntropy, \
        get_right_context_mask, \
        get_positional_signal
except (ModuleNotFoundError, ImportError) as e:
    import model_inputs
    import mrt_utils as mru
    from sampling_utils import SamplingUtils
    import tf_utils
    from transformer_blocks import AttentionBlock, FFNBlock
    from transformer_layers import \
        EmbeddingLayer, \
        MaskedCrossEntropy, \
        get_right_context_mask, \
        get_positional_signal

INT_DTYPE = tf.int32
FLOAT_DTYPE = tf.float32

class Transformer(object):
    """ The main transformer model class. """

    def __init__(self, config):
        # Set attributes
        self.config = config
        self.source_vocab_size = config.source_vocab_sizes[0]
        self.target_vocab_size = config.target_vocab_size
        self.name = 'transformer'

        # Placeholders
        self.inputs = model_inputs.ModelInputs(config)

        # Convert from time-major to batch-major, handle factors
        self.source_ids, \
            self.source_mask, \
            self.target_ids_in, \
            self.target_ids_out, \
            self.target_mask = self._convert_inputs(self.inputs)

        self.training = self.inputs.training
        self.scores = self.inputs.scores
        self.index = self.inputs.index

        # Build the common parts of the graph.
        with tf.compat.v1.name_scope('{:s}_loss'.format(self.name)):
            # (Re-)generate the computational graph
            self.dec_vocab_size = self._build_graph()

        # Build the training-specific parts of the graph.

        with tf.compat.v1.name_scope('{:s}_loss'.format(self.name)):
            # Encode source sequences
            with tf.compat.v1.name_scope('{:s}_encode'.format(self.name)):
                enc_output, cross_attn_mask = self.enc.encode(
                    self.source_ids, self.source_mask)
            # Decode into target sequences
            with tf.compat.v1.name_scope('{:s}_decode'.format(self.name)):
                logits = self.dec.decode_at_train(self.target_ids_in,
                                                  enc_output,
                                                  cross_attn_mask)
            # Instantiate loss layer(s)
            loss_layer = MaskedCrossEntropy(self.dec_vocab_size,
                                            self.config.label_smoothing,
                                            INT_DTYPE,
                                            FLOAT_DTYPE,
                                            time_major=False,
                                            name='loss_layer')
            # Calculate loss
            masked_loss, sentence_loss, batch_loss = \
                loss_layer.forward(logits, self.target_ids_out, self.target_mask, self.training)
            if self.config.print_per_token_pro:
                # e**(-(-log(probability))) =  probability
                self._print_pro = tf.math.exp(-masked_loss)

            sent_lens = tf.reduce_sum(input_tensor=self.target_mask, axis=1, keepdims=False)
            self._loss_per_sentence = sentence_loss * sent_lens
            self._loss = tf.reduce_mean(input_tensor=self._loss_per_sentence, keepdims=False)

            # calculate expected risk
            if self.config.loss_function == 'MRT':
                # self._loss_per_sentence is negative log probability of the output sentence, each element represents
                # the loss of each sample pair.
                self._risk = mru.mrt_cost(self._loss_per_sentence, self.scores, self.index, self.config)

            self.sampling_utils = SamplingUtils(config)


    def _build_graph(self):
        """ Defines the model graph. """
        with tf.compat.v1.variable_scope('{:s}_model'.format(self.name)):
            # Instantiate embedding layer(s)
            if not self.config.tie_encoder_decoder_embeddings:
                enc_vocab_size = self.source_vocab_size
                dec_vocab_size = self.target_vocab_size
            else:
                assert self.source_vocab_size == self.target_vocab_size, \
                    'Input and output vocabularies should be identical when tying embedding tables.'
                enc_vocab_size = dec_vocab_size = self.source_vocab_size

            encoder_embedding_layer = EmbeddingLayer(enc_vocab_size,
                                                     self.config.embedding_size,
                                                     self.config.state_size,
                                                     FLOAT_DTYPE,
                                                     name='encoder_embedding_layer')
            if not self.config.tie_encoder_decoder_embeddings:
                decoder_embedding_layer = EmbeddingLayer(dec_vocab_size,
                                                         self.config.embedding_size,
                                                         self.config.state_size,
                                                         FLOAT_DTYPE,
                                                         name='decoder_embedding_layer')
            else:
                decoder_embedding_layer = encoder_embedding_layer

            if not self.config.tie_encoder_decoder_embeddings:
                softmax_projection_layer = EmbeddingLayer(dec_vocab_size,
                                                          self.config.embedding_size,
                                                          self.config.state_size,
                                                          FLOAT_DTYPE,
                                                          name='softmax_projection_layer')
            else:
                softmax_projection_layer = decoder_embedding_layer

            # Instantiate the component networks
            self.enc = TransformerEncoder(self.config,
                                          encoder_embedding_layer,
                                          self.training,
                                          'encoder')
            self.dec = TransformerDecoder(self.config,
                                          decoder_embedding_layer,
                                          softmax_projection_layer,
                                          self.training,
                                          'decoder')

        return dec_vocab_size

    @property
    def loss_per_sentence(self):
        return self._loss_per_sentence

    @property
    def loss(self):
        return self._loss

    @property
    def risk(self):
        return self._risk

    @property
    def print_pro(self):
        return self._print_pro

    def _convert_inputs(self, inputs):
        # Convert from time-major to batch-major. Note that we take factor 0
        # from x and ignore any other factors.
        source_ids = tf.transpose(a=inputs.x[0], perm=[1,0])
        source_mask = tf.transpose(a=inputs.x_mask, perm=[1,0])
        target_ids_out = tf.transpose(a=inputs.y, perm=[1,0])
        target_mask = tf.transpose(a=inputs.y_mask, perm=[1,0])

        # target_ids_in is a bit more complicated since we need to insert
        # the special <GO> symbol (with value 1) at the start of each sentence
        max_len, batch_size = tf.shape(input=inputs.y)[0], tf.shape(input=inputs.y)[1]
        go_symbols = tf.fill(value=1, dims=[1, batch_size])
        tmp = tf.concat([go_symbols, inputs.y], 0)
        tmp = tmp[:-1, :]
        target_ids_in = tf.transpose(a=tmp, perm=[1,0])
        return (source_ids, source_mask, target_ids_in, target_ids_out,
                target_mask)


class TransformerEncoder(object):
    """ The encoder module used within the transformer model. """

    def __init__(self,
                 config,
                 embedding_layer,
                 training,
                 name):
        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.training = training
        self.name = name

        # Track layers
        self.encoder_stack = dict()
        self.is_final_layer = False

        # Create nodes
        self._build_graph()

    def _embed(self, index_sequence):
        """ Embeds source-side indices to obtain the corresponding dense tensor representations. """
        # Embed input tokens
        return self.embedding_layer.embed(index_sequence)

    def _build_graph(self):
        """ Defines the model graph. """
        # Initialize layers
        with tf.compat.v1.variable_scope(self.name):

            if self.config.transformer_dropout_embeddings > 0:
                self.dropout_embedding = tf.keras.layers.Dropout(rate=self.config.transformer_dropout_embeddings)
            else:
                self.dropout_embedding = None

            for layer_id in range(1, self.config.transformer_enc_depth + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.transformer_enc_depth:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.transformer_ffn_hidden_size, self.config.state_size]
                with tf.compat.v1.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = AttentionBlock(self.config,
                                                     FLOAT_DTYPE,
                                                     self_attention=True,
                                                     training=self.training)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         FLOAT_DTYPE,
                                         is_final=self.is_final_layer,
                                         training=self.training)

                # Maintain layer-wise dict entries for easier data-passing (may change later)
                self.encoder_stack[layer_id] = dict()
                self.encoder_stack[layer_id]['self_attn'] = self_attn_block
                self.encoder_stack[layer_id]['ffn'] = ffn_block

    def encode(self, source_ids, source_mask):
        """ Encodes source-side input tokens into meaningful, contextually-enriched representations. """

        def _prepare_source():
            """ Pre-processes inputs to the encoder and generates the corresponding attention masks."""
            # Embed
            source_embeddings = self._embed(source_ids)
            # Obtain length and depth of the input tensors
            _, time_steps, depth = tf_utils.get_shape_list(source_embeddings)
            # Transform input mask into attention mask
            inverse_mask = tf.cast(tf.equal(source_mask, 0.0), dtype=FLOAT_DTYPE)
            attn_mask = inverse_mask * -1e9
            # Expansion to shape [batch_size, 1, 1, time_steps] is needed for compatibility with attention logits
            attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 1), 1)
            # Differentiate between self-attention and cross-attention masks for further, optional modifications
            self_attn_mask = attn_mask
            cross_attn_mask = attn_mask
            # Add positional encodings
            positional_signal = get_positional_signal(time_steps, depth, FLOAT_DTYPE)
            source_embeddings += positional_signal
            # Apply dropout
            if self.dropout_embedding is not None:
                source_embeddings = self.dropout_embedding(source_embeddings, training=self.training)
            return source_embeddings, self_attn_mask, cross_attn_mask

        with tf.compat.v1.variable_scope(self.name):
            # Prepare inputs to the encoder, get attention masks
            enc_inputs, self_attn_mask, cross_attn_mask = _prepare_source()
            # Propagate inputs through the encoder stack
            enc_output = enc_inputs
            for layer_id in range(1, self.config.transformer_enc_depth + 1):
                enc_output, _ = self.encoder_stack[layer_id]['self_attn'].forward(enc_output, None, self_attn_mask)
                enc_output = self.encoder_stack[layer_id]['ffn'].forward(enc_output)
        return enc_output, cross_attn_mask


class TransformerDecoder(object):
    """ The decoder module used within the transformer model. """

    def __init__(self,
                 config,
                 embedding_layer,
                 softmax_projection_layer,
                 training,
                 name,
                 from_rnn=False):

        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.softmax_projection_layer = softmax_projection_layer
        self.training = training
        self.name = name
        self.from_rnn = from_rnn

        # If the decoder is used in a hybrid system, adjust parameters accordingly
        self.time_dim = 0 if from_rnn else 1

        # Track layers
        self.decoder_stack = dict()
        self.is_final_layer = False

        # Create nodes
        self._build_graph()

    def _embed(self, index_sequence):
        """ Embeds target-side indices to obtain the corresponding dense tensor representations. """
        return self.embedding_layer.embed(index_sequence)

    def _build_graph(self):
        """ Defines the model graph. """
        # Initialize layers
        with tf.compat.v1.variable_scope(self.name):

            if self.config.transformer_dropout_embeddings > 0:
                self.dropout_embedding = tf.keras.layers.Dropout(rate=self.config.transformer_dropout_embeddings)
            else:
                self.dropout_embedding = None

            for layer_id in range(1, self.config.transformer_dec_depth + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.transformer_dec_depth:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.transformer_ffn_hidden_size, self.config.state_size]
                with tf.compat.v1.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = AttentionBlock(self.config,
                                                     FLOAT_DTYPE,
                                                     self_attention=True,
                                                     training=self.training)
                    cross_attn_block = AttentionBlock(self.config,
                                                      FLOAT_DTYPE,
                                                      self_attention=False,
                                                      training=self.training,
                                                      from_rnn=self.from_rnn)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         FLOAT_DTYPE,
                                         is_final=self.is_final_layer,
                                         training=self.training)

                # Maintain layer-wise dict entries for easier data-passing (may change later)
                self.decoder_stack[layer_id] = dict()
                self.decoder_stack[layer_id]['self_attn'] = self_attn_block
                self.decoder_stack[layer_id]['cross_attn'] = cross_attn_block
                self.decoder_stack[layer_id]['ffn'] = ffn_block

    def decode_at_train(self, target_ids, enc_output, cross_attn_mask):
        """ Returns the probability distribution over target-side tokens conditioned on the output of the encoder;
         performs decoding in parallel at training time. """

        def _decode_all(target_embeddings):
            """ Decodes the encoder-generated representations into target-side logits in parallel. """
            # Propagate inputs through the encoder stack
            dec_output = target_embeddings
            for layer_id in range(1, self.config.transformer_dec_depth + 1):
                dec_output, _ = self.decoder_stack[layer_id]['self_attn'].forward(dec_output, None, self_attn_mask)
                dec_output, _ = \
                    self.decoder_stack[layer_id]['cross_attn'].forward(dec_output, enc_output, cross_attn_mask)
                dec_output = self.decoder_stack[layer_id]['ffn'].forward(dec_output)
            return dec_output

        def _prepare_targets():
            """ Pre-processes target token ids before they're passed on as input to the decoder
            for parallel decoding. """
            # Embed target_ids
            target_embeddings = self._embed(target_ids)
            target_embeddings += positional_signal
            if self.dropout_embedding is not None:
                target_embeddings = self.dropout_embedding(target_embeddings, training=self.training)
            return target_embeddings

        def _decoding_function():
            """ Generates logits for target-side tokens. """
            # Embed the model's predictions up to the current time-step; add positional information, mask
            target_embeddings = _prepare_targets()
            # Pass encoder context and decoder embeddings through the decoder
            dec_output = _decode_all(target_embeddings)
            # Project decoder stack outputs and apply the soft-max non-linearity
            full_logits = self.softmax_projection_layer.project(dec_output)
            return full_logits

        with tf.compat.v1.variable_scope(self.name):
            # Transpose encoder information in hybrid models
            if self.from_rnn:
                enc_output = tf.transpose(a=enc_output, perm=[1, 0, 2])
                cross_attn_mask = tf.transpose(a=cross_attn_mask, perm=[3, 1, 2, 0])

            self_attn_mask = get_right_context_mask(tf.shape(input=target_ids)[-1])
            positional_signal = get_positional_signal(tf.shape(input=target_ids)[-1],
                                                      self.config.embedding_size,
                                                      FLOAT_DTYPE)
            logits = _decoding_function()
        return logits
