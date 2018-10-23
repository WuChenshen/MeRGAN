import numpy
import scipy.io as sio

import os
import urllib
import gzip
import cPickle as pickle

import numpy as np
from copy import deepcopy


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import display

# svhn imshow convenience function -- input is a 1D array of length 784
def svhn_imshow(img,im_size):
    plt.imshow(img.reshape([im_size,im_size]), cmap="gray")
    plt.axis('off')

# return a new svhn dataset w/ pixels randomly permuted
def permute_svhn(svhn):
    perm_inds = range(svhn.train.images.shape[1])
    np.random.shuffle(perm_inds)
    print perm_inds
    svhn2 = deepcopy(svhn)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(svhn2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return svhn2

# return a new svhn dataset w/ pixels randomly permuted
def disjoint_svhn(svhn,nums):
    pos_train = []
    for i in range(len(nums)):
        tmp = np.where(svhn.train.labels == nums[i])[0]
        pos_train = np.hstack((pos_train,tmp))
        pos_train = np.asarray(pos_train).astype(int)
        np.random.shuffle(pos_train)
    pos_test = []
    for i in range(len(nums)):
        tmp = np.where(svhn.test.labels == nums[i])[0]
        pos_test = np.hstack((pos_test,tmp))
        pos_test = np.asarray(pos_test).astype(int)
        np.random.shuffle(pos_test)
    pos=[]

    pos.append(pos_train)
    pos.append(pos_test)
    
    svhn2 = lambda:0
    svhn2.train = lambda:0
    svhn2.test = lambda:0
    
    svhn2.train.images = svhn.train.images[pos[0]]
    svhn2.test.images = svhn.test.images[pos[1]]
    svhn2.train.labels = svhn.train.labels[pos[0]]
    svhn2.test.labels = svhn.test.labels[pos[1]]
    return svhn2

# load MNIST dataset with added padding so is 32x32
def load_svhn_32x32(data_path, verbose=True):

    train_data = sio.loadmat(data_path + 'train_32x32.mat')
    train_images = train_data['X'].transpose([3,2,0,1])/255.   # from HWCN to NCHW
    train_labels = train_data['y'].squeeze()
    train_labels[train_labels==10]=0  # Change label 10 to 0 for digit 0

    test_data = sio.loadmat(data_path + 'test_32x32.mat')
    test_images = test_data['X'].transpose([3,2,0,1])/255.   # from HWCN to NCHW
    test_labels = test_data['y'].squeeze()
    test_labels[test_labels==10]=0  # Change label 10 to 0 for digit 0

    # we make sure that the sizes are correct
    assert(len(train_images) == len(train_labels))
    assert(len(test_images) == len(test_labels))
    # print info
    if verbose:
        print("Training Set:   {} samples".format(len(train_images)))
        print("Test Set:       {} samples".format(len(test_images)))
        print("Original Image Shape: {}".format(train_images[0].shape))
    
    # this is a trick to create an empty object,
    # which is shorter than creating a Class with a pass and so on...
    svhn = lambda:0
    svhn.train = lambda:0
    svhn.test = lambda:0
    # and we remake the structure as the original one
    svhn.train.images = train_images
    svhn.test.images = test_images
    svhn.train.labels = train_labels
    svhn.test.labels = test_labels
    
    return svhn

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

def svhn_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data
    #images = images.transpose([0,3,1,2])
    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], targets[i*batch_size:(i+1)*batch_size])

    return get_epoch

def load(batch_size, nums, n_labelled=None, data_dir = 'dataset/svhn/'):

    svhn = load_svhn_32x32(data_dir)
    svhn1 = disjoint_svhn(svhn, nums)

    return (
        svhn_generator((svhn1.train.images,svhn1.train.labels), batch_size, n_labelled), 
        svhn_generator((svhn1.test.images,svhn1.test.labels), batch_size, n_labelled)
    )
