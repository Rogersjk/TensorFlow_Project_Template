"""
@author: Junkai Sun
@file: normalization.py
@time: 2019/3/21 20:15
"""

import numpy as np


class Normalization(object):
    def __init__(self, method):
        self.method = method

    def transform(self, data, train_size):
        self.mean = np.mean(data[:train_size])
        self.std = np.std(data[:train_size])
        self.min = np.min(data[:train_size])
        self.max = np.max(data[:train_size])
        self.interval = self.max - self.min
        if self.method == "std":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            # interval[np.where(interval == 0)] = 1
            return (data - self.min) / self.interval
        elif self.method == "minmax2":
            # interval[np.where(interval == 0)] = 1
            data = (data - self.min) / self.interval
            return data * 2 - 1
        else:
            return data

    def inverse_transform(self, data, transform_data=True):
        # transform_data is to restore data, otherwise, transform loss only restore loss
        if self.method == "std":
            return data * self.std + self.mean if transform_data else data * self.std
        elif self.method == "minmax":
            return data * self.interval + self.min if transform_data else data * self.interval
        elif self.method == "minmax2":
            return (data+1) * self.interval / 2 + self.min if transform_data else data * self.interval / 2
        else:
            return data