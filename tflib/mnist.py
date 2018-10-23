import numpy

import os
import urllib
import gzip
import cPickle as pickle

import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import display

# mnist imshow convenience function -- input is a 1D array of length 784
def mnist_imshow(img,im_size):
    plt.imshow(img.reshape([im_size,im_size]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    print perm_inds
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

# return a new mnist dataset w/ pixels randomly permuted
def disjoint_mnist(mnist,nums):
    pos_train = []
    for i in range(len(nums)):
        tmp = np.where(mnist.train.labels == nums[i])[0]
        pos_train = np.hstack((pos_train,tmp))
        pos_train = np.asarray(pos_train).astype(int)
        np.random.shuffle(pos_train)
    pos_validation = []
    for i in range(len(nums)):
        tmp = np.where(mnist.validation.labels == nums[i])[0]
        pos_validation = np.hstack((pos_validation,tmp))
        pos_validation = np.asarray(pos_validation).astype(int)
        np.random.shuffle(pos_validation)
    pos_test = []
    for i in range(len(nums)):
        tmp = np.where(mnist.test.labels == nums[i])[0]
        pos_test = np.hstack((pos_test,tmp))
        pos_test = np.asarray(pos_test).astype(int)
        np.random.shuffle(pos_test)
    pos=[]
    pos.append(pos_train)
    pos.append(pos_validation)
    pos.append(pos_test)
    
    mnist2 = lambda:0
    mnist2.train = lambda:0
    mnist2.validation = lambda:0
    mnist2.test = lambda:0
    
    mnist2.train.images = mnist.train.images[pos[0]]
    mnist2.validation.images = mnist.validation.images[pos[1]]
    mnist2.test.images = mnist.test.images[pos[2]]
    mnist2.train.labels = mnist.train.labels[pos[0]]
    mnist2.validation.labels = mnist.validation.labels[pos[1]]
    mnist2.test.labels = mnist.test.labels[pos[2]]
    return mnist2

# load MNIST dataset with added padding so is 32x32
def load_mnist_32x32(data_dir, verbose=True):
    # mnist data is by default 28x28 so we add a padding to make it 32x32
    data = input_data.read_data_sets(data_dir, one_hot=False, reshape=False)
    # data cannot be directly modified because it has no set() attribute,
    # so we need to make a copy of it on other variables
    X_trn, y_trn = data.train.images, data.train.labels
    X_val, y_val = data.validation.images, data.validation.labels
    X_tst, y_tst = data.test.images, data.test.labels
    # we make sure that the sizes are correct
    assert(len(X_trn) == len(y_trn))
    assert(len(X_val) == len(y_val))
    assert(len(X_tst) == len(y_tst))
    # print info
    if verbose:
        print("Training Set:   {} samples".format(len(X_trn)))
        print("Validation Set: {} samples".format(len(X_val)))
        print("Test Set:       {} samples".format(len(X_tst)))
        print("Labels: {}".format(y_trn))
        print("Original Image Shape: {}".format(X_trn[0].shape))
    # Pad images with 0s
    X_trn = np.pad(X_trn, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_tst = np.pad(X_tst, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    if verbose:
        print("Updated Image Shape: {}".format(X_trn[0].shape))
    
    # this is a trick to create an empty object,
    # which is shorter than creating a Class with a pass and so on...
    mnist = lambda:0
    mnist.train = lambda:0
    mnist.validation = lambda:0
    mnist.test = lambda:0
    # and we remake the structure as the original one
    mnist.train.images = X_trn
    mnist.validation.images = X_val
    mnist.test.images = X_tst
    mnist.train.labels = y_trn
    mnist.validation.labels = y_val
    mnist.test.labels = y_tst
    
    return mnist

# get equally distributed samples among given classes from a split
def get_ed_samples(data, samples=10):
    # retrieve number of samples for each class
    indx = []
    classes = np.unique(data.labels)
    for cl in range(len(classes)):
        tmp = np.where(data.labels == classes[cl])[0]
        np.random.shuffle(tmp)
        indx = np.hstack((indx,tmp[0:np.min(samples, len(tmp))]))
        indx = np.asarray(indx).astype(int)
        
    return indx

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data
    images = images.transpose([0,3,1,2])
    
    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], targets[i*batch_size:(i+1)*batch_size])

    return get_epoch

def load(batch_size, nums, n_labelled=None, data_dir = 'dataset/mnist'):
    mnist = load_mnist_32x32(data_dir)
    mnist1 = disjoint_mnist(mnist, nums)

    return (
        mnist_generator((mnist1.train.images,mnist1.train.labels), batch_size, n_labelled), 
        mnist_generator((mnist1.validation.images,mnist1.validation.labels), batch_size, n_labelled), 
        mnist_generator((mnist1.test.images,mnist1.test.labels), batch_size, n_labelled)
    )
