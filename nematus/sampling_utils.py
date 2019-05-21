import numpy
import tensorflow as tf
import logging

class SamplingUtils(object):
    def __init__(self, config_or_settings_obj):
        self.sampling_temperature = config_or_settings_obj.sampling_temperature
        self.translation_strategy = config_or_settings_obj.translation_strategy

    def adjust_logits(self, logits):
        if self.sampling_temperature != 1.0:
            logging.debug("adjust temperature")
            logits = logits / tf.constant(self.sampling_temperature, dtype=tf.float32)

        return logits


