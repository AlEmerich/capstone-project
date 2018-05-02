from .board import Board
import tensorflow as tf


class TensorBoard(Board):
    def __init(self, tf_session):
        self.tf_session = tf_session
        super(TensorBoard, self).__init__()

    def on_launch(self, **kwargs):
        labels = kwargs["labels"]

        for k, v in kwargs.items():
            tf.summary.scalar(k, v)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.folder,
                                            self.tf_session.graph)

    def on_running(self, **kwargs):
        ydatas = kwargs['ydatas']
        xdata = kwargs['xdata']
        info = kwargs['info']

        summary = self.tf_session.run(self.merged)
        self.writer.add_summary(summary)

    def on_reset(self):
        pass
