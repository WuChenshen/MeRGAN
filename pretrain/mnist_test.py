
import inspect
import os


import numpy as np
import tensorflow as tf
import time

from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Mlp:
    def __init__(self, batch_size = 64, vgg16_npy_path=None, trainable=True, dropout=0.5, class_num = 10, image_size = 32):
        self.var_dict = {}
        self.class_num = class_num  
	self.trainable = trainable
	self.batch_size = batch_size
        self.data_dict = None
        self.image_size = image_size
    def build(self, rgb, train_mode = None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        print("build model started")
        rgb = tf.reshape(rgb, [-1, self.image_size*self.image_size]) 

        self.fc1 = self.fc_layer(rgb, self.image_size*self.image_size, 1024, "fc1")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu1 = tf.nn.relu(self.fc1)

        self.fc2 = self.fc_layer(self.relu1, 1024, 512, "fc2")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu2 = tf.nn.relu(self.fc2)

        self.fc3 = self.fc_layer(self.relu2, 512, 256, "fc3")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu3 = tf.nn.relu(self.fc3)

        self.fc4 = self.fc_layer(self.relu3, 256, 128, "fc4")
        self.relu4 = tf.nn.relu(self.fc4)

        self.fc5 = self.fc_layer(self.relu4, 128, 10, "fc5")

        self.prob = tf.nn.softmax(self.fc5, name="prob")


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            #var = tf.Variable(value, name=var_name)
	    var = tf.get_variable(name="%s"%var_name,initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
