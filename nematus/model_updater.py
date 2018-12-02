import numpy
import tensorflow as tf

"""Helper class for training using multiple GPUs.

Given a set of model replicas and an optimizer, it takes care of splitting
minibatches, feeding them to the individual replicas, and then combining and
applying the resulting gradients (and losses).
"""
class ModelUpdater(object):
    def __init__(self, config, num_gpus, replicas, optimizer, global_step,
                 summary_writer=None):
        assert len(replicas) > 0
        assert (len(replicas) == num_gpus
                or (len(replicas) == 1 and num_gpus == 0))

        self.replicas = replicas
        self.global_step = global_step
        self.summary_writer = summary_writer

        self.replica_weights = []
        for i in range(len(self.replicas)):
            name = 'replica_weight_{}'.format(i)
            placeholder = tf.placeholder(name=name, shape=(), dtype=tf.float32)
            self.replica_weights.append(placeholder)

        weighted_losses = []
        all_grad_vars = []
        for i in range(len(self.replicas)):
            device_type = "GPU" if num_gpus > 0 else "CPU"
            device_spec = tf.DeviceSpec(device_type=device_type, device_index=i)
            with tf.device(device_spec):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i>0)):
                    loss = self._regularize(replicas[i].loss, config.decay_c,
                                            config.map_decay_c)
                    gradients = optimizer.compute_gradients(loss)
                    all_grad_vars.append(gradients)
                    weight = self.replica_weights[i]
                    weighted_losses.append(loss*weight)

        self.loss = sum(weighted_losses) / sum(self.replica_weights)

        grad_vars = self._average_gradients(all_grad_vars, self.replica_weights)
        grads, varss = zip(*grad_vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(
            grads, clip_norm=config.clip_c)
        # Might be interesting to see how the global norm changes over time,
        # attach a summary?
        grad_vars = zip(clipped_grads, varss)
        self.apply_grads = optimizer.apply_gradients(
            grad_vars, global_step=self.global_step)

        tf.summary.scalar(name='mean_cost', tensor=self.loss)
        tf.summary.scalar(name='t', tensor=self.global_step)
        self.merged = tf.summary.merge_all()

    def _regularize(self, loss, decay_c, map_decay_c):
        with tf.variable_scope("loss"):
            # Optionally, add an L2 loss term.
            if decay_c > 0.0:
                l2_loss = tf.constant(0.0, dtype=tf.float32)
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * tf.constant(decay_c, dtype=tf.float32)
                loss += l2_loss
            # Optionally, add an L2 loss term based on a prior model.
            if map_decay_c > 0.0:
                map_l2_loss = tf.constant(0.0, dtype=tf.float32)
                map_l2_acc = []
                for v in tf.trainable_variables():
                    prior_name = 'prior/'+v.name.split(':')[0]
                    prior_v = tf.get_variable(
                        prior_name, initializer=v.initialized_value(),
                        trainable=False, collections=['prior_variables'],
                        dtype=v.initialized_value().dtype)
                    map_l2_acc.append(tf.nn.l2_loss(v - prior_v))
                map_l2_loss = tf.add_n(map_l2_acc) * tf.constant(map_decay_c, dtype=tf.float32)
                loss += map_l2_loss
        return loss

    def update(self, session, x, x_mask, y, y_mask, target_lang, write_summary):
        """Update the model for a single minibatch."""
        batch_size = x.shape[-1]
        assert batch_size > 0

        num_replicas = len(self.replicas)

        # Split the minibatch into sub-minibatches, one per replica. If the
        # minibatch contains fewer items than there are replicas, then the
        # last sub-minibatches will be empty, which requires special handling.
        split_x = numpy.array_split(x, num_replicas, axis=-1)
        split_x_mask = numpy.array_split(x_mask, num_replicas, axis=-1)
        split_y = numpy.array_split(y, num_replicas, axis=-1)
        split_y_mask = numpy.array_split(y_mask, num_replicas, axis=-1)

        feed_dict = {}
        for i in range(num_replicas):
            # If the sub-minibatch is empty then replace it with a non-empty
            # dummy one.
            is_dummy = (i >= batch_size)
            assert (split_x[i].shape[-1] == 0) == is_dummy
            weight = 0.0 if is_dummy else float(split_x[i].shape[-1])
            feed_dict[self.replica_weights[i]] = weight
            j = 0 if is_dummy else i
            feed_dict[self.replicas[i].inputs.x] = split_x[j]
            feed_dict[self.replicas[i].inputs.x_mask] = split_x_mask[j]
            feed_dict[self.replicas[i].inputs.y] = split_y[j]
            feed_dict[self.replicas[i].inputs.y_mask] = split_y_mask[j]
            feed_dict[self.replicas[i].inputs.target_lang_id] = target_lang
            feed_dict[self.replicas[i].inputs.training] = True

        out = [self.global_step, self.apply_grads, self.loss]
        if write_summary:
            out += [self.merged]
        out_values = session.run(out, feed_dict=feed_dict)
        if write_summary:
            assert self.summary_writer != None
            self.summary_writer.add_summary(out_values[3], out_values[0])
        loss = out_values[2] * batch_size
        return loss

    def _average_gradients(self, all_grad_vars, replica_weights):
        # all_grad_vars is a list of lists of tuples.  The outer list contains
        # one list for each replica. Each inner list is the optimizer's
        # grad_vars list for that replica. Each replica has an associated
        # weight (to allow for replicas receiving batches of different sizes).

        normalized_weights = []
        for w in replica_weights:
            normalized_weights.append(w / sum(replica_weights))

        # Create a dictionary mapping from each variable name to a list of
        # (grad, var) pairs (one pair from each replica).
        d = {}
        for grad_vars in all_grad_vars:
            for g, v in grad_vars:
                if v.name not in d:
                    d[v.name] = []
                d[v.name].append((g, v))
        # For each variable, average the gradients from all replicas and store
        # the result in avg_grad_vars.
        avg_grad_vars = []
        for var_name, gv_list in d.items():
            var = gv_list[0][1]
            found_none_value = False
            for g, v in gv_list:
                if g == None:
                    found_none_value = True
                    break
            if found_none_value:
                avg_grad_vars.append((None, var))
            else:
                weighted_grads = []
                for i, (g, v) in enumerate(gv_list):
                    assert v == var
                    expanded = tf.expand_dims(g * normalized_weights[i], 0)
                    weighted_grads.append(expanded)
                tmp = tf.concat(axis=0, values=weighted_grads)
                avg_grad = tf.reduce_sum(tmp, 0)
                avg_grad_vars.append((avg_grad, var))
        return avg_grad_vars
