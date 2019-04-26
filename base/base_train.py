from tqdm import tqdm
import numpy as np
import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data_loader, config, logger):
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
        # save best model on validation set
        self.min_loss = float("inf")
        self.best_epoch = 0

        self.X_train, self.Y_train = self.data_loader.get_train_samples()
        self.X_eval, self.Y_eval = self.data_loader.get_eval_samples()
        self.X_test, self.Y_test = self.data_loader.get_test_samples()
        self.input_state_size = self.X_train.shape[0]

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch=None):
        if self.config.drop_batch_remainder:
            num_iter_per_epoch = self.input_state_size // self.config.batch_size
        else:
            num_iter_per_epoch = int(np.ceil(self.input_state_size / self.config.batch_size))
        iterations = tqdm(range(num_iter_per_epoch))
        for _ in iterations:
            x_batch, y_batch = self.sess.run(self.data_loader.next_batch)
            self.sess.run(self.model.train_step, feed_dict={self.model.x: x_batch, self.model.y: y_batch})

        train_loss = self.sess.run(self.model.loss,
                                   feed_dict={self.model.x: self.X_eval, self.model.y: self.Y_eval})
        # early stopping
        if epoch - self.best_epoch > self.config.early_stopping:
            print("Early stopping at epoch %d, best epoch is %d" % (epoch, self.best_epoch))
            return 0
        # save best model on validation set
        eval_loss = self.sess.run(self.model.loss,
                                  feed_dict={self.model.x: self.X_eval, self.model.y: self.Y_eval})
        if eval_loss < self.min_loss:
            self.min_loss = eval_loss
            self.best_epoch = epoch
            print("Epoch %d get a better model, model loss is %f" % (self.best_epoch, self.min_loss))
            self.model.save(self.sess)
        summary_dict = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        self.logger.summarize(epoch, summaries_dict=summary_dict)