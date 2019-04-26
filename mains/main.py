"""
@author: Junkai Sun
@file: main.py
@time: 2018/12/5 19:57
"""
import numpy as np
# np.random.seed(35)
import tensorflow as tf
# tf.set_random_seed(35)
import sys
import os
import h5py
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.split(os.path.realpath(__file__))[0][:-6])
from datetime import datetime
from data_loader.data_generator import DataGenerator
from models.S2S_TAtt import S2S_TAtt
from trainers.S2STrainer import S2STrainer
from tools.config import process_config
from tools.utils import create_dirs
from tools.logger import Logger

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('DIR', 'logs', 'default DIR')
flags.DEFINE_string('params', '', 'arg params')
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_string("model_name", "S2S_TAtt", "model selection")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("num_epochs", 500, "number of epochs")
flags.DEFINE_integer("len_closeness", 8, "length of closeness steps")
flags.DEFINE_string('dataset', 'BikeNYC', 'datasets choice')

flags.DEFINE_integer("train", 1, "whether need to train")
flags.DEFINE_integer("tensorboard", 0, "whether log in tensorboard")
flags.DEFINE_integer("use_t_att", 1, "whether use temporal attention")

flags.DEFINE_string("activation", "relu", "activation function for dense layer")
flags.DEFINE_string("optimizer", "Adam", "optimizer for training process")
flags.DEFINE_string("normalize", "std", "normalization method")

models = {"S2S_TAtt": S2S_TAtt}

def compute_keyarea_loss(file_path):
    with h5py.File(file_path, "r") as f:
        ks2 = list(f.keys())
        res2 = []
        for i in ks2:
            res2.append(f[i].value)
    weights = res2[2]
    print(np.mean(weights, axis=1))
    real = res2[0]
    pred = res2[1].reshape(-1, 4, 2, 16, 8)
    print("RMSE is %f, MAE is %f" % (np.sqrt(np.mean(np.square(real - pred))),
                                     np.mean(np.abs(real - pred))))
    key_index = np.array([[7, 3],
                          [6, 3],
                          [6, 2],
                          [9, 0],
                          [3, 4],
                          [5, 1]])
    masked_real, masked_pred = [], []
    for i in key_index:
        masked_real.append(real[:, :, :, i[0], i[1]])
        masked_pred.append(pred[:, :, :, i[0], i[1]])
    masked_real = np.stack(masked_real, axis=-1)
    masked_pred = np.stack(masked_pred, axis=-1)
    print(masked_real.shape)
    print("maksed RMSE is %f, masked MAE is %f" % (
    np.sqrt(np.mean(np.square(masked_real - masked_pred))), np.mean(np.abs(masked_real - masked_pred))))

def main():
    tmp = FLAGS.params   # necessary to get updated FLAGS values
    config = process_config("../configs/config.json", FLAGS.flag_values_dict())
    create_dirs([config.summary_dir, config.checkpoint_dir])
    print(config)
    sess = tf.Session()
    # Data process and generator
    print("Data Start Loading!", datetime.now())
    data = DataGenerator(config)
    print("Data Loaded!", datetime.now())
    # Create model
    # model_name = FLAGS.model_name
    model = S2S_TAtt(data.handle, config, data.next_batch)
    print("Model Initialized!", datetime.now())
    if config.tensorboard == 1:
        # Tensorboard logger
        logger = Logger(sess, config)
        print("Logger Initialized!", datetime.now())
        # Model trainer
        trainer = S2STrainer(sess, model, data, config, logger)
    else:
        trainer = S2STrainer(sess, model, data, config)
    print("Trainer Initialized!", datetime.now())
    # Training processs
    if FLAGS.train:
        trainer.train()
    with h5py.File("./result.h5", "w") as f:
        f.create_dataset("predictions", data=trainer.predictions, compression="lzf")
        f.create_dataset("groundtruth", data=trainer.ground_truth, compression="lzf")
        f.create_dataset("weights", data=trainer.weights, compression="lzf")
    # print(np.mean(trainer.weights, axis=1))


if __name__ == "__main__":
    main()
    compute_keyarea_loss("result.h5")