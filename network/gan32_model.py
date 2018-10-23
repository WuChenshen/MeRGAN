"""WGAN-GP ResNet for CIFAR-10"""
import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import pdb
import time
import functools
#import locale
#locale.setlocale(locale.LC_ALL, '')

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?

N_COLOR = 1
IMAGE_SIZE = 32
DIM_G = 32 # Generator dimensionality
DIM_D = 32 # Critic dimensionality
OUTPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * N_COLOR # Number of pixels in CIFAR10 (32*32*3)

def GeneratorAndDiscriminator(n_color):
    return functools.partial(Generator, n_color = n_color), functools.partial(Discriminator, n_color = n_color)

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)            
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Generator(name, n_samples, labels, noise=None, n_color = 3):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear(name + 'Generator.Input', 128, 4*4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])
    output = Normalize(name + 'Generator.BN1', output, labels=labels)
    #output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN1', [0], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.2', 4*DIM_G, 2*DIM_G, 5, output)
    output = Normalize(name + 'Generator.BN2', output, labels=labels)
    #output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.3', 2*DIM_G, DIM_G, 5, output)
    output = Normalize(name + 'Generator.BN3', output, labels=labels)
    #output = lib.ops.batchnorm.Batchnorm(name + 'Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D(name + 'Generator.5', DIM_G, n_color, 5, output)

    output = tf.tanh(output)
    return tf.reshape(output, [-1, IMAGE_SIZE * IMAGE_SIZE * n_color])


def Discriminator(inputs, labels=None, name = '', n_color = 3):
    output = tf.reshape(inputs, [-1, n_color, IMAGE_SIZE, IMAGE_SIZE])
    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.1', n_color, DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(name + 'Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output_wgan = lib.ops.linear.Linear(name + 'Discriminator.Output', 4*4*4*DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    output_acgan = lib.ops.linear.Linear(name + 'Discriminator.ACGANOutput', 4*4*4*DIM_D, 10, output)
    return output_wgan, output_acgan
