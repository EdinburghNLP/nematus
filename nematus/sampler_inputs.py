import tensorflow as tf


class SamplerInputs:
    """Input placeholders for RandomSampler and BeamSearchSampler."""

    def __init__(self):

        # Number of sentences in the input. When sampling, this is not
        # necessarily the same as the batch size, hence the modified name. The
        # actual batch size (i.e. as seen by the model) will vary: usually
        # it's batch_size_x * beam_size because we tile the input sentences,
        # but in the Transformer encoder it's just batch_size_x.
        self.batch_size_x = tf.compat.v1.placeholder(
            name='batch_size_x',
            shape=(),
            dtype=tf.int32)

        # Maximum translation length.
        self.max_translation_len = tf.compat.v1.placeholder(
            name='max_translation_len',
            shape=(),
            dtype=tf.int32)

        # Alpha parameter for length normalization.
        self.normalization_alpha = tf.compat.v1.placeholder(
            name='normalization_alpha',
            shape=(),
            dtype=tf.float32)
