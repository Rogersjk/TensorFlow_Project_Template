"""
@author: Junkai Sun
@file: logger.py
@time: 2018/12/29 12:07
"""
import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        self.summary_placeholders[tag] = tf.placeholder('float32', (), name=tag)
                        self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()