import os

import numpy as np
import tensorflow as tf

from ..env import OUTPUT_DIR

# -----------------------------------------------------------------------------

LOG_DIR = os.path.join(OUTPUT_DIR, 'log')

# -----------------------------------------------------------------------------


class SummaryGroup(object):
    def __init__(self, manager, key):
        self.manager = manager
        self.writer = tf.summary.FileWriter(
            os.path.join(LOG_DIR, key),
            manager.sess.graph,
        )

        self.reset()

    def reset(self):
        self.batch_sizes = []
        self.batch_values = {}

    def add_batch(self, batch_size, values):
        if self.batch_sizes:
            assert (
                frozenset(values.keys()) ==
                frozenset(self.batch_values.keys())
            )
        else:
            for name in values.keys():
                self.batch_values[name] = []

        self.batch_sizes.append(batch_size)
        for name, value in values.items():
            self.batch_values[name].append(value)

    def write(self, global_step):
        batch_sizes = np.array(self.batch_sizes)
        fetches = []
        feed_dict = {}
        summary_values = {}

        for name, values in self.batch_values.items():
            summary_runner = self.manager.get_summary_runner(name)
            epoch_value = np.average(values, weights=batch_sizes)
            fetches.append(summary_runner.summary)
            feed_dict[summary_runner.placeholder] = epoch_value
            summary_values[name] = epoch_value

        epoch_summaries = self.manager.sess.run(fetches, feed_dict=feed_dict)
        for epoch_summary in epoch_summaries:
            self.writer.add_summary(epoch_summary, global_step)

        self.writer.flush()
        self.reset()
        return summary_values


class SummaryRunner(object):
    def __init__(
        self,
        manager,
        name,
        summary_op=tf.summary.scalar,
        summary_collections=('managed_summaries',),
        placeholder_dtype=tf.float32,
        placeholder_shape=(),
        **kwargs
    ):
        self.manager = manager
        self.placeholder = tf.placeholder(
            placeholder_dtype, shape=placeholder_shape,
        )
        self.summary = summary_op(
            name,
            self.placeholder,
            collections=summary_collections,
            **kwargs
        )

    def run(self, value):
        return self.manager.sess.run(self.summary, feed_dict={
            self.placeholder: value,
        })


class SummaryManager(object):
    def __init__(self, sess=None):
        self._sess = sess

        self.groups = {}
        self.summary_runners = {}

    @property
    def sess(self):
        return self._sess or tf.get_default_session()

    def __getitem__(self, key):
        if key not in self.groups:
            self.groups[key] = SummaryGroup(self, key)
        return self.groups[key]

    def get_summary_runner(self, name, **kwargs):
        if name not in self.summary_runners:
            self.summary_runners[name] = SummaryRunner(self, name, **kwargs)
        return self.summary_runners[name]

    def write(self, global_step):
        summary_values = {}
        for key, group in self.groups.items():
            summary_values[key] = group.write(global_step)
        return summary_values
