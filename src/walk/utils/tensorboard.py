from .board import Board
import tensorflow as tf


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

        with tf.variable_scope("plot"):
            # Set placeholders to dict and use it as scalar summary
            for label in self.labels:
                self.d[label] = tf.Variable(0.0)
                tf.summary.scalar(label, self.d[label])

            # Set the placeholder for the info text
            self.d["info"] = tf.placeholder(tf.string, name="info")
            tf.summary.text("info", self.d["info"])

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

        # Construct the dict we will feed with
        feed = {}
        for i, label in enumerate(self.labels):
            feed[self.d[label]] = ydatas[i]
            feed[self.d["info"]] = "" if not info else info

        self.merged = tf.summary.merge_all()
        # Get the value to plot
        summary = self.tf_session.run(self.merged, feed)
        # Write to disk
        self.writer.add_summary(summary, xdata)

    def on_reset(self, t):
        """No need to reset with tensorboard.
        """
        pass

    def save(self):
        """No need to save anything with tensorboard.
        """
        pass
