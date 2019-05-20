import numpy
import tensorflow as tf
import logging

class SamplingUtils(object):
    def __init__(self, config_or_settings_obj):
        self.sampling_temperature = config_or_settings_obj.sampling_temperature
        self.sampling_truncation_prob = config_or_settings_obj.sampling_truncation_prob
        self.translation_strategy = config_or_settings_obj.translation_strategy

    def adjust_logits(self, logits):
        if self.sampling_temperature != 1.0:
            logging.debug("adjust temperature")
            logits = logits / tf.constant(self.sampling_temperature, dtype=tf.float32)

#        if self.sampling_truncation_prob < 1.0:
#            logging.debug("truncate")
#            probs = tf.nn.softmax(logits, axis=-1)
#            sorted_prob_idxs = tf.argsort(probs, axis=-1, direction='DESCENDING')
#            sorted_probs = tf.gather(probs, sorted_prob_idxs)
#            cum_probs = tf.cumsum(sorted_probs, axis=-1)
#            mask = tf.less_equal(cum_probs, tf.constant(self.sampling_truncation_prob, dtype=tf.float32))
#            masked_sorted_probs = sorted_probs * tf.cast(mask, tf.float32)
#            inv_idxs = 
#            masked_probs = tf.gather(masked_sorted_probs, tf.invert_permutation(sorted_prob_idxs))
#            normalized_probs = masked_probs / tf.reduce_sum(masked_probs, axis=-1, keep_dims=True)
#            logits = tf.log(normalized_probs)

        return logits


