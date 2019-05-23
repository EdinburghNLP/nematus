"""Adapted from Nematode: https://github.com/demelin/nematode """

import tensorflow as tf
import numpy

import model_inputs
from transformer_layers import \
    EmbeddingLayer, \
    MaskedCrossEntropy, \
    get_shape_list, \
    get_right_context_mask, \
    get_positional_signal
from transformer_blocks import AttentionBlock, FFNBlock
from transformer_inference import greedy_search, beam_search

from sampling_utils import SamplingUtils

class Transformer(object):
    """ The main transformer model class. """

    def __init__(self, config):
        # Set attributes
        self.config = config
        self.source_vocab_size = config.source_vocab_sizes[0]
        self.target_vocab_size = config.target_vocab_size
        self.name = 'transformer'
        self.int_dtype = tf.int32
        self.float_dtype = tf.float32

        # Placeholders
        self.inputs = model_inputs.ModelInputs(config)

        # Convert from time-major to batch-major, handle factors
        self.source_ids, \
            self.source_mask, \
            self.target_ids_in, \
            self.target_ids_out, \
            self.target_mask = self._convert_inputs(self.inputs)

        self.training = self.inputs.training

        # Build the common parts of the graph.
        with tf.name_scope('{:s}_loss'.format(self.name)):
            # (Re-)generate the computational graph
            self.dec_vocab_size = self._build_graph()

        # Build the training-specific parts of the graph.

        with tf.name_scope('{:s}_loss'.format(self.name)):
            # Encode source sequences
            with tf.name_scope('{:s}_encode'.format(self.name)):
                enc_output, cross_attn_mask = self.enc.encode(
                    self.source_ids, self.source_mask)
            # Decode into target sequences
            with tf.name_scope('{:s}_decode'.format(self.name)):
                logits = self.dec.decode_at_train(self.target_ids_in,
                                                  enc_output,
                                                  cross_attn_mask)
            # Instantiate loss layer(s)
            loss_layer = MaskedCrossEntropy(self.dec_vocab_size,
                                            self.config.label_smoothing,
                                            self.int_dtype,
                                            self.float_dtype,
                                            time_major=False,
                                            name='loss_layer')
            # Calculate loss
            masked_loss, sentence_loss, batch_loss = \
                loss_layer.forward(logits, self.target_ids_out, self.target_mask, self.training)

            sent_lens = tf.reduce_sum(self.target_mask, axis=1, keepdims=False)
            self._loss_per_sentence = sentence_loss * sent_lens
            self._loss = tf.reduce_mean(self._loss_per_sentence, keepdims=False)
        
        self.sampling_utils = SamplingUtils(config)

    def _build_graph(self):
        """ Defines the model graph. """
        with tf.variable_scope('{:s}_model'.format(self.name)):
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
                                                     self.float_dtype,
                                                     name='encoder_embedding_layer')
            if not self.config.tie_encoder_decoder_embeddings:
                decoder_embedding_layer = EmbeddingLayer(dec_vocab_size,
                                                         self.config.embedding_size,
                                                         self.config.state_size,
                                                         self.float_dtype,
                                                         name='decoder_embedding_layer')
            else:
                decoder_embedding_layer = encoder_embedding_layer

            if not self.config.tie_encoder_decoder_embeddings:
                softmax_projection_layer = EmbeddingLayer(dec_vocab_size,
                                                          self.config.embedding_size,
                                                          self.config.state_size,
                                                          self.float_dtype,
                                                          name='softmax_projection_layer')
            else:
                softmax_projection_layer = decoder_embedding_layer

            # Instantiate the component networks
            self.enc = TransformerEncoder(self.config,
                                          encoder_embedding_layer,
                                          self.training,
                                          self.float_dtype,
                                          'encoder')
            self.dec = TransformerDecoder(self.config,
                                          decoder_embedding_layer,
                                          softmax_projection_layer,
                                          self.training,
                                          self.int_dtype,
                                          self.float_dtype,
                                          'decoder')

        return dec_vocab_size

    @property
    def loss_per_sentence(self):
        return self._loss_per_sentence

    @property
    def loss(self):
        return self._loss

    def _convert_inputs(self, inputs):
        # Convert from time-major to batch-major. Note that we take factor 0
        # from x and ignore any other factors.
        source_ids = tf.transpose(inputs.x[0], perm=[1,0])
        source_mask = tf.transpose(inputs.x_mask, perm=[1,0])
        target_ids_out = tf.transpose(inputs.y, perm=[1,0])
        target_mask = tf.transpose(inputs.y_mask, perm=[1,0])

        # target_ids_in is a bit more complicated since we need to insert
        # the special <GO> symbol (with value 1) at the start of each sentence
        max_len, batch_size = tf.shape(inputs.y)[0], tf.shape(inputs.y)[1]
        go_symbols = tf.fill(value=1, dims=[1, batch_size])
        tmp = tf.concat([go_symbols, inputs.y], 0)
        tmp = tmp[:-1, :]
        target_ids_in = tf.transpose(tmp, perm=[1,0])
        return (source_ids, source_mask, target_ids_in, target_ids_out,
                target_mask)


class TransformerEncoder(object):
    """ The encoder module used within the transformer model. """

    def __init__(self,
                 config,
                 embedding_layer,
                 training,
                 float_dtype,
                 name):
        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.training = training
        self.float_dtype = float_dtype
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
        with tf.variable_scope(self.name):
            for layer_id in range(1, self.config.transformer_enc_depth + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.transformer_enc_depth:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.transformer_ffn_hidden_size, self.config.state_size]
                with tf.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = AttentionBlock(self.config,
                                                     self.float_dtype,
                                                     self_attention=True,
                                                     training=self.training)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         self.float_dtype,
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
            _, time_steps, depth = get_shape_list(source_embeddings)
            # Transform input mask into attention mask
            inverse_mask = tf.cast(tf.equal(source_mask, 0.0), dtype=self.float_dtype)
            attn_mask = inverse_mask * -1e9
            # Expansion to shape [batch_size, 1, 1, time_steps] is needed for compatibility with attention logits
            attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 1), 1)
            # Differentiate between self-attention and cross-attention masks for further, optional modifications
            self_attn_mask = attn_mask
            cross_attn_mask = attn_mask
            # Add positional encodings
            positional_signal = get_positional_signal(time_steps, depth, self.float_dtype)
            source_embeddings += positional_signal
            # Apply dropout
            if self.config.transformer_dropout_embeddings > 0:
                source_embeddings = tf.layers.dropout(source_embeddings,
                                                      rate=self.config.transformer_dropout_embeddings, training=self.training)
            return source_embeddings, self_attn_mask, cross_attn_mask

        with tf.variable_scope(self.name):
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
                 int_dtype,
                 float_dtype,
                 name,
                 from_rnn=False):

        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.softmax_projection_layer = softmax_projection_layer
        self.training = training
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
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

    def _get_initial_memories(self, batch_size, beam_size):
        """ Initializes decoder memories used for accelerated inference. """
        initial_memories = dict()
        for layer_id in range(1, self.config.transformer_dec_depth + 1):
            initial_memories['layer_{:d}'.format(layer_id)] = \
                {'keys': tf.tile(tf.zeros([batch_size, 0, self.config.state_size]), [beam_size, 1, 1]),
                 'values': tf.tile(tf.zeros([batch_size, 0, self.config.state_size]), [beam_size, 1, 1])}
        return initial_memories

    def _build_graph(self):
        """ Defines the model graph. """
        # Initialize layers
        with tf.variable_scope(self.name):
            for layer_id in range(1, self.config.transformer_enc_depth + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.transformer_enc_depth:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.transformer_ffn_hidden_size, self.config.state_size]
                with tf.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = AttentionBlock(self.config,
                                                     self.float_dtype,
                                                     self_attention=True,
                                                     training=self.training)
                    cross_attn_block = AttentionBlock(self.config,
                                                      self.float_dtype,
                                                      self_attention=False,
                                                      training=self.training,
                                                      from_rnn=self.from_rnn)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         self.float_dtype,
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
            # Apply input dropout
            dec_input = \
                tf.layers.dropout(target_embeddings, rate=self.config.transformer_dropout_embeddings, training=self.training)
            # Propagate inputs through the encoder stack
            dec_output = dec_input
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
            if self.config.transformer_dropout_embeddings > 0:
                target_embeddings = tf.layers.dropout(target_embeddings,
                                                      rate=self.config.transformer_dropout_embeddings, training=self.training)
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

        with tf.variable_scope(self.name):
            # Transpose encoder information in hybrid models
            if self.from_rnn:
                enc_output = tf.transpose(enc_output, [1, 0, 2])
                cross_attn_mask = tf.transpose(cross_attn_mask, [3, 1, 2, 0])

            self_attn_mask = get_right_context_mask(tf.shape(target_ids)[-1])
            positional_signal = get_positional_signal(tf.shape(target_ids)[-1],
                                                      self.config.embedding_size,
                                                      self.float_dtype)
            logits = _decoding_function()
        return logits
