"""
@author: Junkai Sun
@file: SepRankTrainer.py
@time: 2018/12/29 11:42
"""
# from tqdm import tqdm
import numpy as np
import tensorflow as tf
from datetime import datetime
import sys

class S2STrainer:
    def __init__(self, sess, model, data_loader, config, logger=[]):
        """
        Constructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data_loader: The data loader if specified. (You will find Dataset API example)
        """
        # Assign all class attributes

        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data_loader = data_loader
        self.inverse_transform = self.data_loader.normalize.inverse_transform

        self.ground_truth = self.inverse_transform(self.data_loader.target_test)

        self.train_handle = self.sess.run(self.data_loader.train_iterator.string_handle())
        self.val_handle = self.sess.run(self.data_loader.val_iterator.string_handle())
        self.test_handle = self.sess.run(self.data_loader.test_iterator.string_handle())
        # save best model on validation set
        self.min_loss = float("inf")  # record th minimum validation loss
        self.test_loss = 999   # record the test set loss corresponding to the best validation loss
        self.best_epoch = 0
        self.predictions, self.weights = 0, 0

        self.sample_num = self.data_loader.train_num
        print("Train number is ", self.sample_num)

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
            sys.stdout.flush()
            if epoch - self.best_epoch > self.config.early_stopping:
                print("Early stopping in epoch %d !" % (epoch))
                break
        print("Best epoch is %d, corresponding validation loss is %f, test loss is %f" %
              (self.best_epoch, self.min_loss, self.test_loss))

    def train_epoch(self, epoch=None):
        if self.config.drop_batch_remainder:
            num_iter_per_epoch = self.sample_num // self.config.batch_size
        else:
            num_iter_per_epoch = int(np.ceil(self.sample_num / self.config.batch_size))
        # iterations = tqdm(range(num_iter_per_epoch))
        iterations = range(num_iter_per_epoch)
        losses = []
        # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for _ in iterations:
            _, train_loss=self.sess.run([self.model.train_step, self.model.loss], feed_dict={self.model.handle: self.train_handle})
            losses.append(train_loss)
        train_avg_loss = self.inverse_transform(np.sqrt(np.mean(losses)), False)
        print("%s --> Epoch %d , train loss is %f" %
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, train_avg_loss))
        val_loss_tmp = self.sess.run(self.model.loss, feed_dict={self.model.handle: self.val_handle})
        test_loss_tmp, predictions, weights = self.sess.run([self.model.loss, self.model.predictions, self.model.TAtt_weigths],
                                                            feed_dict={self.model.handle: self.test_handle})
        # val_loss_tmp, test_loss_tmp = self.ratio * np.sqrt(val_loss_tmp), self.ratio * np.sqrt(test_loss_tmp)
        val_loss_tmp, test_loss_tmp = self.inverse_transform(np.array([np.sqrt(val_loss_tmp), np.sqrt(test_loss_tmp)]), False)
        if val_loss_tmp < self.min_loss:
            print("Epoch %d  get a better result, validation loss is %f, test loss is %f" %
                  (epoch, val_loss_tmp, test_loss_tmp))
            self.best_epoch = epoch
            self.min_loss = val_loss_tmp
            self.test_loss = test_loss_tmp
            self.predictions = self.inverse_transform(predictions)
            self.weights = weights
            self.model.save(self.sess)

        if self.config.tensorboard == 1:
            summary_dict = {
                "epoch": epoch,
                "train_loss": train_avg_loss,
            }
            self.logger.summarize(epoch, summaries_dict=summary_dict)
