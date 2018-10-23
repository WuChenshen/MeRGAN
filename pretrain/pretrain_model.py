"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

import logging
import lsun_test
import mnist_test
import svhn_test
import pdb
import os
logging.basicConfig(level = logging.INFO)
class pretrain_classifier(object):
    def __init__(self, sess, image_size=64, batch_size=64, class_num = 10, dataset='lsun'):

        self.sess = sess
        self.class_num = class_num
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        if self.dataset == 'lsun':
            self.checkpoint_dir = 'pretrain/lsun10/lsun10.model-100002'
        elif self.dataset == 'svhn':
            self.checkpoint_dir = 'pretrain/svhn/svhn.model-25654'
        elif self.dataset == 'mnist':
            self.checkpoint_dir = 'pretrain/mnist/mnist.model-10197'
        self.build_model_annotation()
    def build_model_annotation(self):
        if self.dataset == 'lsun':
            self.images = tf.placeholder(tf.float32, shape = [self.batch_size, self.image_size, self.image_size,  3])
            model = lsun_test.Vgg16(batch_size = self.batch_size, class_num =  self.class_num)
        elif self.dataset == 'svhn':
            self.images = tf.placeholder(tf.float32, shape = [self.batch_size, self.image_size, self.image_size,  3])
            model = svhn_test.Conv(batch_size = self.batch_size, class_num =  self.class_num)
        else:
            self.images = tf.placeholder(tf.float32, shape = [self.batch_size, self.image_size, self.image_size])
            model = mnist_test.Mlp(batch_size = self.batch_size, class_num = self.class_num)
        model.build(self.images)
        self.pred_probality = model.prob
        self.pred_cls = tf.cast(tf.argmax(model.prob, dimension=1), tf.int32)
        if self.dataset == 'lsun':
            self.restored_vars = [_var for _var in tf.trainable_variables() if _var.name.startswith('conv') or  _var.name.startswith('fc')]
        elif self.dataset == 'svhn':
            self.restored_vars = [_var for _var in tf.trainable_variables() if _var.name.startswith('svhn')]
        else:
            self.restored_vars = [_var for _var in tf.trainable_variables() if  _var.name.startswith('fc')]
        init_op =tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op) 
        self.saver = tf.train.Saver(self.restored_vars)
        self.saver.restore(self.sess, self.checkpoint_dir)

    def test(self,data):
        counter = 0
        predicted_labels = []
        predicted_probality = []
        for iteration in xrange(len(data) / self.batch_size ):
            _begin = iteration* self.batch_size
            _end = (iteration + 1)* self.batch_size
            _images = data[_begin:_end]
            _pred_cls, _pred_probality = self.sess.run([self.pred_cls, self.pred_probality],feed_dict = {self.images: _images})
            predicted_labels.extend(_pred_cls)
            predicted_probality.extend(_pred_probality)
        return predicted_labels, np.asarray(predicted_probality) 
         # test save
		      
