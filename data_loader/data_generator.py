import numpy as np
# np.random.seed(35)
import tensorflow as tf
# tf.set_random_seed(35)
import h5py
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tools.normalization import Normalization


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.normalize = Normalization(self.config.normalize)
        # load data here
        self.load_data()
        self.handle, self.next_batch, self.train_iterator, \
        self.val_iterator, self.test_iterator = self.build_input_pipelines()


    def get_samples(self, date_str, flows):
        date_target, flow_target = [], []
        date_closeness, flow_closeness = [], []
        date_period, flow_period = [], []
        date_str = np.array([str(int(x)-1) for x in date_str])
        str2time = lambda x: datetime.strptime(x, "%Y%m%d%H")
        date_datetime = [str2time(x) for x in date_str]
        target_steps = self.config.prediction_steps
        for index, date1 in enumerate(date_datetime):
            target_index = range(index, index + target_steps)
            closeness_index = range(index - self.config.len_closeness, index)
            if self.config.len_period == 0:
                period_index = []
            else:
                period_index = np.concatenate([np.array(target_index) - days * self.config.timestamps_one_day for days in range(self.config.len_period, 0, -1)])
            # valid iteration if date is continuous in the range for target, closeness and period
            valid_target = (index + target_steps <= len(date_datetime)) and\
                           (date_datetime[target_index[0]] + timedelta(hours=target_steps-1) == date_datetime[target_index[-1]])
            valid_closeness = (self.config.len_closeness == 0) or (closeness_index[0] >= 0) and \
                              (date_datetime[closeness_index[0]] + timedelta(hours=self.config.len_closeness-1) == date_datetime[closeness_index[-1]])
            valid_period = (self.config.len_period == 0) or (period_index[0] >= 0) and \
                           (date_datetime[period_index[0]] + timedelta(hours=self.config.len_period * self.config.timestamps_one_day) == date1)
            if not (valid_target and valid_closeness and valid_period):
                continue
            date_target.append(np.take(date_str, target_index, axis=0))
            flow_target.append(np.take(flows, target_index, axis=0))
            date_closeness.append(np.take(date_str, closeness_index, axis=0))
            flow_closeness.append(np.take(flows, closeness_index, axis=0))
            date_period.append(np.take(date_str, period_index, axis=0))
            flow_period.append(np.take(flows, period_index, axis=0))
        return np.asarray(flow_closeness), np.asarray(flow_period), np.asarray(flow_target), np.asarray(date_target)

    def load_data(self):
        if self.config.dataset == "BikeNYC":
            data_file = self.config.BikeNYC_File
            # last 10 days as test set
            self.test_size = 10 * self.config.timestamps_one_day - self.config.prediction_steps
        elif self.config.dataset == "TaxiBJ":
            data_file = self.config.TaxiBJ_File
            # last 4 weeks as test set
            self.test_size = 28 * self.config.timestamps_one_day - self.config.prediction_steps
        self.validation_size = 10 * self.config.timestamps_one_day - self.config.prediction_steps
        with h5py.File(data_file, "r") as f:
            # keys = list(f.keys())
            date = f["date"].value.astype(np.str)
            flows = f["data"].value.astype(np.float32)
        flows = self.normalize.transform(flows, flows.shape[0] - (28+10) * self.config.timestamps_one_day)
        closeness, period, target, date_target = self.get_samples(date, flows)

        self.closeness_train, closeness_val_test, self.period_train, period_val_test, self.target_train, target_val_test = train_test_split(
            closeness, period, target, test_size=self.validation_size + self.test_size, shuffle=False)

        self.closeness_val, self.closeness_test, self.period_val, self.period_test, self.target_val, self.target_test = train_test_split(
            closeness_val_test, period_val_test, target_val_test, test_size=self.test_size, shuffle=False)
        self.train_num = self.target_train.shape[0]

    def build_input_pipelines(self):
        train_iterator = tf.data.Dataset.from_tensor_slices({
            "closeness": self.closeness_train,
            "period": self.period_train,
            "target": self.target_train
        }).shuffle(
            self.config.shuffle_buffer_size).batch(self.config.batch_size, drop_remainder=self.config.drop_batch_remainder).prefetch(
            self.config.prefetch_buffer_size).repeat().make_one_shot_iterator()

        val_iterator = tf.data.Dataset.from_tensor_slices({
            "closeness": self.closeness_val,
            "period": self.period_val,
            "target": self.target_val
        }).batch(self.validation_size).repeat().make_one_shot_iterator()

        test_iterator = tf.data.Dataset.from_tensor_slices({
            "closeness": self.closeness_test,
            "period": self.period_test,
            "target": self.target_test
        }).batch(self.test_size).repeat().make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        shapes = train_iterator.output_shapes
        new_shapes = {}
        for k in shapes.keys():
            new_shapes[k] = tf.TensorShape([None] + shapes[k].dims[1:])
        feedable_iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, new_shapes)
        return handle, feedable_iterator.get_next(), train_iterator, val_iterator, test_iterator

    def get_train_samples(self):
        return self.closeness_train, self.period_train, self.target_train

    def get_test_samples(self):
        return self.closeness_test, self.period_test, self.target_test


if __name__ == "__main__":
    from tools.config import process_config
    config = process_config("../configs/config.json")
    data_gen = DataGenerator(config)
    with tf.Session() as sess:
        handle = sess.run(data_gen.train_iterator.string_handle())
        print(handle)
        for i in range(2):
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            data1 = sess.run(data_gen.next_batch, feed_dict={data_gen.handle: handle})
            print(data1[2], data1[3], data1[4])
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))