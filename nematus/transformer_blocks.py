"""Adapted from Nematode: https://github.com/demelin/nematode """

import tensorflow as tf

from transformer_layers import \
    ProcessingLayer, \
    FeedForwardNetwork

from transformer_attention_modules import MultiHeadAttentionLayer


# from attention_modules import SingleHeadAttentionLayer, FineGrainedAttentionLayer


class AttentionBlock(object):
    """ Defines a single attention block (referred to as 'sub-layer' in the paper) comprising of a single multi-head
    attention layer preceded by a pre-processing layer and followed by a post-processing layer. """

    def __init__(self,
                 config,
                 float_dtype,
                 self_attention,
                 training,
                 from_rnn=False,
                 tie_attention=False):
        # Set attributes
        self.self_attention = self_attention
        if not tie_attention:
            if self_attention:
                attn_name = 'self_attn'
            else:
                attn_name = 'cross_attn'
        else:
            attn_name = 'tied_attn'

        memory_size = config.state_size
        if from_rnn:
            memory_size *= 2

        # Build layers
        self.pre_attn = ProcessingLayer(config.state_size,
                                        use_layer_norm=True,
                                        dropout_rate=0.,
                                        training=training,
                                        name='pre_{:s}_sublayer'.format(attn_name))

        self.attn = MultiHeadAttentionLayer(memory_size,
                                            config.state_size,
                                            config.state_size,
                                            config.state_size,
                                            config.state_size,
                                            config.transformer_num_heads,
                                            float_dtype,
                                            dropout_attn=config.transformer_dropout_attn,
                                            training=training,
                                            name='{:s}_sublayer'.format(attn_name))

        self.post_attn = ProcessingLayer(config.state_size,
                                         use_layer_norm=False,
                                         dropout_rate=config.transformer_dropout_residual,
                                         training=training,
                                         name='post_{:s}_sublayer'.format(attn_name))

    def forward(self, inputs, memory_context, attn_mask, layer_memories=None):
        """ Propagates input data through the block. """
        if not self.self_attention:
            assert (memory_context is not None), \
                'Encoder memories have to be provided for encoder-decoder attention computation.'
        attn_inputs = self.pre_attn.forward(inputs)
        attn_outputs, layer_memories = self.attn.forward(attn_inputs, memory_context, attn_mask, layer_memories)
        block_out = self.post_attn.forward(attn_outputs, residual_inputs=inputs)
        return block_out, layer_memories


class FFNBlock(object):
    """ Defines a single feed-forward network block (referred to as 'sub-layer' in the transformer paper) comprising of
    a single feed-forward network preceded by a pre-processing layer and followed by a post-processing layer. """

    def __init__(self,
                 config,
                 ffn_dims,
                 float_dtype,
                 is_final,
                 training):
        # Set attributes
        self.is_final = is_final

        # Build layers
        self.pre_ffn = ProcessingLayer(config.state_size,
                                       use_layer_norm=True,
                                       dropout_rate=0.,
                                       training=training,
                                       name='pre_ffn_sublayer')
        self.ffn = FeedForwardNetwork(ffn_dims,
                                      float_dtype,
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      use_layer_norm=False,
                                      dropout_rate=config.transformer_dropout_relu,
                                      training=training,
                                      name='ffn_sublayer')
        self.post_ffn = ProcessingLayer(config.state_size,
                                        use_layer_norm=False,
                                        dropout_rate=config.transformer_dropout_residual,
                                        training=training,
                                        name='post_ffn_sublayer')
        if is_final:
            self.pre_final = ProcessingLayer(config.state_size,
                                             use_layer_norm=True,
                                             dropout_rate=0.,
                                             training=training,
                                             name='final_transform')

    def forward(self, inputs):
        """ Propagates input data through the block. """
        ffn_inputs = self.pre_ffn.forward(inputs)
        ffn_outputs = self.ffn.forward(ffn_inputs)
        block_out = self.post_ffn.forward(ffn_outputs, residual_inputs=inputs)
        if self.is_final:
            block_out = self.pre_final.forward(block_out)
        return block_out
