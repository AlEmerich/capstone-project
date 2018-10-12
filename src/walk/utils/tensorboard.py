from .board import Board
import tensorflow as tf
import numpy as np


class TensorBoard(Board):
    """Extends Board class to use tensorboard to plot metrics
    in order to be used the same way we use matplotlib.
    """
    def __init__(self, tf_session, name_run):
        """Set the session, the scope and call
        the super constructor.
        """
        self.tf_session = tf_session
        super(TensorBoard, self).__init__(name_run)

    def on_launch(self, **kwargs):
        """Instantiate tensorflow placeholders and variables
        which will receive data to plot.
        :param labels: label of the value to plot.
        """
        self.labels = kwargs["labels"]
        # The dict we will have to feed
        self.d = {}
        self.on_launch_summary = []
        self.on_reset_summary = []

        with tf.variable_scope("plot"):
            # Set placeholders to dict and use it as scalar summary
            for label in self.labels:
                self.d[label] = tf.Variable(0.0)
                summary = tf.summary.scalar(label, self.d[label])
                self.on_launch_summary.append(summary)

            # Set the placeholder for the info text
            self.d["info"] = tf.placeholder(tf.string, name="info")
            summary = tf.summary.text("info", self.d["info"])
            self.on_launch_summary.append(summary)

            # Variables for plotting rewards for each episode
            self.r_var = tf.Variable(0.0)
            summary = tf.summary.scalar(
                "Reward_per_episod", self.r_var)
            self.on_reset_summary.append(summary)

            self.avg_r_var = tf.Variable(0.0)
            summary = tf.summary.scalar("Average_reward_per_episod",
                                        self.avg_r_var)
            self.on_reset_summary.append(summary)

        # Instantiate the writer
        self.writer = tf.summary.FileWriter(self.path,
                                            self.tf_session.graph)

    def on_running(self, **kwargs):
        """Feed placeholder and variables with values provided.

        :param ydatas: the value to feed in. Must be in the
        same order than labels list provided in on_launch.

        :param xdata: the xdata to plot in x axis (basically t)

        :param info: random text to show.
        """
        ydatas = kwargs['ydatas']
        xdata = kwargs['xdata']
        info = kwargs['info']
        additional = kwargs['additional']

        # Construct the dict we will feed with
        feed = {}
        for i, label in enumerate(self.labels):
            feed[self.d[label]] = ydatas[i]
            feed[self.d["info"]] = "" if not info else info

        merged = tf.summary.merge(self.on_launch_summary)
        # Get the value to plot
        summary = self.tf_session.run(merged, feed)
        # Write to disk
        self.writer.add_summary(summary, xdata)

        if additional:
            merge_additional = tf.summary.merge(additional)
            summary_additional = self.tf_session.run(merge_additional)
            self.writer.add_summary(summary_additional, xdata)

    def on_reset(self, t, rewards):
        """No need to reset with tensorboard.
        """
        merge = tf.summary.merge(self.on_reset_summary)
        summary = self.tf_session.run(merge, {
            self.r_var: sum(rewards),
            self.avg_r_var: np.average(rewards)})
        self.writer.add_summary(summary, t)

    def save(self):
        """No need to save anything with tensorboard.
        """
        pass
