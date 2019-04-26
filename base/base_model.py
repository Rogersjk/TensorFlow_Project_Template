import tensorflow as tf
class BaseModel:
    def __init__(self, config):
        self.config = config
        # save attribute .. NOTE DON'T FORGET TO CONSTRUCT THE SAVER ON YOUR MODEL
        self.model_path = self.config.checkpoint_dir+self.config.model_name + self.config.params + ".ckpt"
        self.saver = None

    def dense_layers(self, inputs, fcs=[]):
        for fc in fcs:
            inputs = fc(inputs)
        return inputs

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        # print("Saving model...")
        self.saver.save(sess, self.model_path)
        # print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        self.saver.restore(sess, self.model_path)
        print("Model loaded")

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.train.Saver()

    def build_model(self):
        raise NotImplementedError

    def predict(self, x):
        pass