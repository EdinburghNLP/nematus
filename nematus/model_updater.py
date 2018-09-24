import numpy
import tensorflow as tf

"""Helper class for model training.

Currently, ModelUpdater just gathers together the code for processing a
minibatch and updating the model. It will soon be replaced with a multiple
GPU version.
"""
class ModelUpdater(object):
    def __init__(self, config, model, optimizer, global_step,
                 summary_writer=None):
        self.model = model
        self.global_step = global_step
        self.summary_writer = summary_writer

        self.objective = self.model.get_objective()
        grad_vars = optimizer.compute_gradients(self.objective)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(
            grads, clip_norm=config.clip_c)
        # Might be interesting to see how the global norm changes over time,
        # attach a summary?
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = optimizer.apply_gradients(
            grad_vars, global_step=self.global_step)

        tf.summary.scalar(name='mean_cost', tensor=self.objective)
        tf.summary.scalar(name='t', tensor=self.global_step)
        self.merged = tf.summary.merge_all()

    def update(self, session, x, x_mask, y, y_mask, target_lang, write_summary):
        """Update the model for a single minibatch."""
        batch_size = x.shape[-1]
        assert batch_size > 0

        feed_dict = {}
        feed_dict[self.model.inputs.x] = x
        feed_dict[self.model.inputs.x_mask] = x_mask
        feed_dict[self.model.inputs.y] = y
        feed_dict[self.model.inputs.y_mask] = y_mask
        feed_dict[self.model.inputs.target_lang_id] = target_lang
        feed_dict[self.model.inputs.training] = True

        out = [self.global_step, self.apply_grads, self.objective]
        if write_summary:
            out += [self.merged]
        out_values = session.run(out, feed_dict=feed_dict)
        if write_summary:
            assert self.summary_writer != None
            self.writer.add_summary(out_values[3], out_values[0])
        loss = out_values[2] * batch_size
        return loss
