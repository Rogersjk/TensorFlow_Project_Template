"""
@author: Junkai Sun
@file: SepRank.py
@time: 2018/12/5 19:58
"""
import numpy as np
import tensorflow as tf
from base.base_model import BaseModel

map_activation = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "leaky_relu": tf.nn.leaky_relu}
map_optimizer = {"Adam": tf.train.AdamOptimizer, "RMSProp": tf.train.RMSPropOptimizer}

class S2S_TAtt(BaseModel):
    def __init__(self, handle, config, next_batch=None):
        super().__init__(config)
        self.handle = handle
        self.num_hidden = self.config.num_hidden
        self.TAtt_weigths = []
        self.attention_size = self.config.attention_size
        self.next_batch = next_batch
        self.activation = map_activation[self.config.activation]
        self.optimizer = map_optimizer[self.config.optimizer]
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.name_scope("Input_Module"):

            self.closeness, self.period, self.target = self.next_batch["closeness"], \
                                                       self.next_batch["period"], \
                                                       self.next_batch["target"]
            sample_size = tf.shape(self.closeness)[0]
            sp1 = self.closeness.get_shape().as_list()
            sp2 = self.period.get_shape().as_list()
            sp3 = self.target.get_shape().as_list()
            # self.target = tf.reshape(self.target, (-1, sp3[1], np.prod(sp3[2:])))
            # reshape the image-formatted flow to vector (batch_size, 2, 16, 8) --> (batch_size, 2*16*8)
            self.closeness, self.period, self.target = tf.reshape(self.closeness, (-1, sp1[1], np.prod(sp1[2:]))), \
                                                       tf.reshape(self.period, (-1, sp2[1], np.prod(sp2[2:]))), \
                                                       tf.reshape(self.target, (-1, sp3[1], np.prod(sp3[2:])))
            '''
            # first convolution and then reshape to vector
            conv1 = tf.layers.Conv2D(filters=32, kernel_size=(3, 3),  padding="same",
                                     activation=tf.nn.relu)
            conv2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
            flatten = tf.layers.Flatten()
            fc1 = tf.layers.Dense(256)
            self.closeness = tf.transpose(tf.map_fn(lambda x: fc1(flatten(conv2(conv1(x)))),
                                                    tf.transpose(self.closeness, (1, 0, 4, 2, 3))), (1, 0, 2))
            '''


        with tf.variable_scope("Encoder_Module", reuse=tf.AUTO_REUSE):
            encoder_cells = tf.nn.rnn_cell.LSTMCell(self.num_hidden, initializer=tf.orthogonal_initializer, state_is_tuple=True)
            state = encoder_cells.zero_state(sample_size, tf.float32)
            encoder_states = []
            for i in range(self.config.len_closeness):
                encoder_output, state = encoder_cells(self.closeness[:, i], state)
                encoder_states.append(encoder_output)

        with tf.variable_scope("Decoder_Module", reuse=tf.AUTO_REUSE):
            decoder_cells = tf.nn.rnn_cell.LSTMCell(self.num_hidden, initializer=tf.orthogonal_initializer, state_is_tuple=True)
            state = decoder_cells.zero_state(sample_size, tf.float32)
            hidden_state = state.h
            flow_input = self.closeness[:, -1]
            output_dim = np.prod(sp1[2:])
            out_layer = tf.layers.Dense(output_dim, name="decoder_output_to_prediction")
            predictions = []
            for j in range(self.config.prediction_steps):
                if self.config.use_t_att:
                    context = self.attention(hidden_state, encoder_states, j)
                else:
                    context = hidden_state
                # context = hidden_state
                decoder_input = tf.concat([flow_input, context], axis=1)
                decoder_output, state = decoder_cells(decoder_input, state)
                predictions.append(out_layer(decoder_output))
                flow_input = predictions[-1]
                hidden_state = state.h
            self.predictions = tf.stack(predictions, axis=1)

        with tf.name_scope("Optimizer"):
            self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.predictions)
            self.train_step = self.optimizer(self.config.learning_rate).minimize(self.loss)

    def attention(self, decoder_state, encoder_states, k):
        merged_states = []
        for encoder_state in encoder_states:
            merged_states.append(tf.concat([decoder_state, encoder_state], axis=1))
        merged_states = tf.stack(merged_states, axis=1)
        hidden_size = merged_states.get_shape().as_list()[-1]
        if k == 0:
            self.w_omega = tf.get_variable("w_omega", [hidden_size, self.attention_size], tf.float32)
            self.b_omega = tf.get_variable("b_omega", [self.attention_size], tf.float32)
            self.u_omega = tf.get_variable("u_omega", [self.attention_size, ], tf.float32)
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.nn.tanh(tf.tensordot(merged_states, self.w_omega, axes=1) + self.b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        self.TAtt_weigths.append(alphas)

        # Output of RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(tf.stack(encoder_states, axis=1) * tf.expand_dims(alphas, -1), 1)
        # output = tf.reduce_sum(tf.stack(encoder_states, axis=1) * alphas, 1)
        return output
