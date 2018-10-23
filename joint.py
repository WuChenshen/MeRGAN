from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.getcwd())

import pdb
import time
import functools

import pickle
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.plot
import tflib.save_images

import tflib.mnist
import tflib.svhn

import network.gan32_model
from pretrain.pretrain_model import pretrain_classifier

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--result_path', default='', type=str, help='Result Path')
parser.add_argument('--dataset', default='mnist', type=str, help='Dataset: svhn , mnist')      
parser.add_argument('--iters', default=0, type=int, help='Number of iterations per task')

parser.add_argument('--test', action = 'store_true', help='Test mode')

args = parser.parse_args()

if args.test:
    RESULT_DIR = args.result_path
else:
    RESULT_DIR = 'result/' + args.result_path 
SAMPLES_DIR = os.path.join(RESULT_DIR, 'samples/')
MODEL_DIR = os.path.join(RESULT_DIR, 'model/')
DATASET = args.dataset # lsun10, mnist, svhn

if DATASET == 'mnist':
    N_PIXELS = 32
    N_COLORS = 1
    ITERS = 40001 if args.iters == 0 else args.iters
    NUM_CLASS = 10
    all_classes = [0,1,2,3,4,5,6,7,8,9]
    DATASET_DIR = 'dataset/mnist'
elif DATASET == 'svhn':
    N_PIXELS = 32
    N_COLORS = 3
    ITERS = 80001 if args.iters == 0 else args.iters
    NUM_CLASS = 10
    all_classes = [0,1,2,3,4,5,6,7,8,9]
    DATASET_DIR = 'dataset/svhn'

OUTPUT_DIM = N_PIXELS * N_PIXELS * N_COLORS # Number of pixels in each iamge
N_GPUS = 1 # Number of GPUs

# LOG
SAVE_SAMPLES_STEP = 200 # Generate and save samples every SAVE_SAMPLES_STEP
CHECKPOINT_STEP = 4000


# ACGAN
ACGAN = True
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1. # How to scale generator's ACGAN loss relative to WGAN loss

# WGAN-GP
CRITIC_ITERS = 5 # How many iterations to train the critic for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
D_LR = 0.0001
G_LR = 0.0001
BETA1_D = 0.0
BETA1_G = 0.0

BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS

# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(MODEL_DIR):
  print("*** create checkpoint dir %s" % MODEL_DIR)
  os.makedirs(MODEL_DIR)

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if DATASET == 'mnist' or DATASET == 'svhn':
    Generator, Discriminator = network.gan32_model.GeneratorAndDiscriminator(N_COLORS)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_COLORS, N_PIXELS, N_PIXELS])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    n_samples = BATCH_SIZE//len(DEVICES)
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs, disc_real_acgan_costs, disc_fake_acgan_costs = [],[],[],[]
    disc_acgan_real_accs, disc_acgan_fake_accs = [], []
    for device_index, (device, real_data_conv, real_labels) in enumerate(zip(DEVICES, split_real_data_conv, labels_splits)):
        with tf.device(device):
            real_data = tf.reshape(2*(real_data_conv-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            fake_labels_splits = tf.cast(tf.random_uniform([BATCH_SIZE//len(DEVICES)])*NUM_CLASS, tf.int32)
            fake_data = Generator('New.', BATCH_SIZE//len(DEVICES), labels = fake_labels_splits)

            disc_real, disc_real_acgan = Discriminator(real_data)
            disc_fake, disc_fake_acgan = Discriminator(fake_data)

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_wgan = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(shape=[BATCH_SIZE//len(DEVICES),1], minval=0., maxval=1. )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates)[0], interpolates)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_wgan_pure = disc_wgan
            disc_wgan += LAMBDA*gradient_penalty
            disc_cost = disc_wgan

            if ACGAN:
                disc_real_acgan_costs.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real_acgan, labels=real_labels)))
                disc_fake_acgan_costs.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels_splits)))
                disc_cost += ACGAN_SCALE * tf.add_n(disc_real_acgan_costs)
                gen_cost += ACGAN_SCALE_G * tf.add_n(disc_fake_acgan_costs)
                disc_acgan_real_accs.append(tf.reduce_mean(
                    tf.cast(tf.equal(tf.to_int32(tf.argmax(disc_real_acgan, dimension=1)), real_labels ), tf.float32)))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(tf.equal(tf.to_int32(tf.argmax(disc_fake_acgan, dimension=1)), fake_labels_splits ), tf.float32)))

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)
    if ACGAN:
        disc_acgan_real_acc = tf.add_n(disc_acgan_real_accs) / len(DEVICES)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)    

    # For generating samples
    fixed_noise = tf.constant(np.tile(np.random.normal(size=(10,1, 128)).astype('float32'),[1,10,1]).reshape(100,128))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator('New.', 100, labels = fixed_labels, noise=fixed_noise)
    def generate_image(name):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        samples = samples.reshape((100, N_COLORS, N_PIXELS, N_PIXELS))
        if DATASET == 'mnist':
            samples = np.tile(samples,[1,3,1,1])

        lib.save_images.save_images(samples, SAMPLES_DIR+'samples_{}.png'.format(name))

    ckpt_saver = tf.train.Saver(max_to_keep=10000)
    session.run(tf.global_variables_initializer())

    if DATASET == 'mnist':
        train_gen, dev_gen, _ = lib.mnist_disjoint.load(BATCH_SIZE, all_classes, data_dir = DATASET_DIR)
    elif DATASET == 'svhn':
        train_gen, dev_gen = lib.svhn_disjoint.load(BATCH_SIZE, all_classes, data_dir = DATASET_DIR)

    def inf_train_gen():
        while True:
            for (images,labels) in train_gen():
                yield images,labels
    gen = inf_train_gen()

    # Save a batch of ground-truth samples
    _x,_ = inf_train_gen().next()
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
    _x_r = ((_x_r+1.)*(255.99//2)).astype('int32')
    if DATASET == 'lsun10' or DATASET == 'svhn':
        lib.save_images.save_images(_x_r.reshape((BATCH_SIZE//N_GPUS, N_COLORS, N_PIXELS, N_PIXELS)), '%s/samples_groundtruth_%s.png' % (SAMPLES_DIR, 4))
    elif DATASET == 'mnist':
        lib.save_images.save_images(np.tile(_x_r.reshape([BATCH_SIZE//N_GPUS,N_COLORS,N_PIXELS,N_PIXELS]),[1,3,1,1]), '%s/samples_groundtruth_%s.png' % (SAMPLES_DIR, 4))


    if args.test:
        labels_np = np.array([0,1,2,3,4,5,6,7,8,9]*10)
        BATCH_SIZE_TEST = 100
        model = pretrain_classifier(session, image_size = N_PIXELS, batch_size = BATCH_SIZE_TEST, dataset = DATASET)
        if DATASET == 'mnist' or DATASET == 'svhn':
            task_test = [4,9]       # Evaluate models after task 4 (5th) and task9 (10th)

        for task in task_test:
            # Load Model
            LOAD_MODEL_FILE = MODEL_DIR + 'WGAN_GP.model' + '-' + str(ITERS - 1)
            ckpt_saver.restore(session, LOAD_MODEL_FILE)

            # Evaluate accuracy
            acc = 0.
            repeat = 100
            samples = Generator('New.', BATCH_SIZE_TEST, labels = fixed_labels)
            for i in range(repeat):
                _samples = session.run(samples) * 128 + 128
                _samples = _samples.astype(int).reshape([BATCH_SIZE_TEST,N_COLORS,N_PIXELS,N_PIXELS]).transpose([0,2,3,1]).squeeze()
                predicted_labels, predicted_probality = model.test(_samples)
                _acc = float(np.sum(np.equal(predicted_labels,labels_np))) / BATCH_SIZE_TEST
                acc += _acc/repeat
            print('Task: {}   All classes accuracy: {}'.format(task+1, acc))
        exit()


    for iteration in range(ITERS):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _images, _labels = gen.next()
            _gen_cost, _disc_fake_acgan_costs, _ = session.run([gen_cost, disc_fake_acgan_costs, gen_train_op], 
                                                        feed_dict = {all_real_labels: _labels})
            lib.plot.plot('g-cost', _gen_cost)
            lib.plot.plot('acgan-fake', np.mean(_disc_fake_acgan_costs))

        # Train critic
        for i in range(CRITIC_ITERS):
            _images, _labels = gen.next()
            _disc_cost, _disc_wgan, _disc_wgan_pure, _disc_real_acgan_costs,_disc_acgan_real_acc,_disc_acgan_fake_acc, _ = session.run(
                [disc_cost, disc_wgan, disc_wgan_pure,  disc_real_acgan_costs,disc_acgan_real_acc,disc_acgan_fake_acc, disc_train_op], 
                    feed_dict={all_real_data_conv: _images, all_real_labels: _labels})

        lib.plot.plot('d-cost', _disc_cost)
        lib.plot.plot('wgan-pure', _disc_wgan_pure)
        lib.plot.plot('penalty', _disc_wgan - _disc_wgan_pure)
        if ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan-real', np.mean(_disc_real_acgan_costs))
            lib.plot.plot('real_acc', _disc_acgan_real_acc)
            lib.plot.plot('fake_acc', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % SAVE_SAMPLES_STEP == 0 or iteration in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]:                
            generate_image(str(iteration))

        # Save checkpoint
        if iteration % CHECKPOINT_STEP == 0:# or iteration in [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]:
            ckpt_saver.save(session, MODEL_DIR + 'WGAN_GP.model', iteration)
        if iteration < 10 or iteration % 100 == 0:
            lib.plot.flush(path = RESULT_DIR)
        lib.plot.tick()

