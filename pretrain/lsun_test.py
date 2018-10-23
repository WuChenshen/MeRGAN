import inspect
import os


import numpy as np
import tensorflow as tf
import time

from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, batch_size = 64, class_num = 10):
        self.class_num = class_num  
	self.batch_size = batch_size

    def build(self, rgb,train_mode = None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        print("build model started")
        rgb_scaled = rgb 

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
       # assert red.get_shape().as_list()[1:] == [224, 224, 1]
       # assert green.get_shape().as_list()[1:] == [224, 224, 1]
       # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        assert red.get_shape().as_list()[1:] == [64, 64, 1]
        assert green.get_shape().as_list()[1:] == [64, 64, 1]
        assert blue.get_shape().as_list()[1:] == [64, 64, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [64, 64, 3]

        conv1_1 = self.conv_layer(bgr,3,64, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1,64,64, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1,64,128, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1,128,128, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2,128,256, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1,256,256, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2,256,256, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3,256,512, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1,512,512, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2,512,512, "conv4_3")
        #pool4, arg4 = tf.nn.max_pool_with_argmax(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4,512,512, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1,512,512, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, 512,512,"conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')
	pool5_len = reduce(lambda x, y: x*y, pool5.get_shape().as_list()[1:])

        self.fc6 = self.fc_layer(pool5, pool5_len, 1024, "fc6_lsun10")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, 1024, 512, "fc7_lsun10")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, 512, 10, "fc8_lsun10")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

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
        var = tf.get_variable(name="%s"%var_name,initializer=initial_value)
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

