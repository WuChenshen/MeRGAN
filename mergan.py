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
import tflib.lsun


import network.gan64_model
import network.gan32_model
from pretrain.pretrain_model import pretrain_classifier

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--result_path', default='', type=str, help='Result Path')
parser.add_argument('--load_model', default='', type=str, help='Load Model')

parser.add_argument('--RA', action = 'store_true', help='Alignment Path')
parser.add_argument('--RA_factor', default = 1e-4, type = float, help='Alignment Rate')

parser.add_argument('--JTR', action = 'store_true', help='Joint Training with Replay')

parser.add_argument('--dataset', default='mnist', type=str, help='Dataset: mnist, svhn, lsun')

parser.add_argument('--iters', default=0, type=int, help='Number of iterations per task')

parser.add_argument('--test', action = 'store_true', help='Test mode')

args = parser.parse_args()

if args.test:
    RESULT_DIR = args.result_path
else:
    RESULT_DIR = 'result/' + args.result_path 
SAMPLES_DIR = os.path.join(RESULT_DIR, 'samples/')
MODEL_DIR = os.path.join(RESULT_DIR, 'model/')

DATASET = args.dataset
if DATASET == 'lsun':
    N_PIXELS = 64
    N_COLORS = 3
    ITERS = 20001 if args.iters == 0 else args.iters
    NUM_CLASS = 4
    all_classes = ['bedroom','kitchen','church_outdoor','tower']
    REDUCE_BATCH = False
    DATASET_DIR = 'dataset/lsun'
elif DATASET == 'mnist':
    N_PIXELS = 32
    N_COLORS = 1
    ITERS = 4001 if args.iters == 0 else args.iters
    NUM_CLASS = 10
    all_classes = [0,1,2,3,4,5,6,7,8,9]
    REDUCE_BATCH = True
    DATASET_DIR = 'dataset/mnist'
elif DATASET == 'svhn':
    N_PIXELS = 32
    N_COLORS = 3
    ITERS = 8001 if args.iters == 0 else args.iters
    NUM_CLASS = 10
    all_classes = [0,1,2,3,4,5,6,7,8,9]
    REDUCE_BATCH = True
    DATASET_DIR = 'dataset/svhn'

RA_FACTOR = args.RA_factor
RA = args.RA
JTR = args.JTR

if args.load_model != '':
    LOAD_MODEL = True
    LOAD_MODEL_FILE = args.load_model
else:
    LOAD_MODEL = False

# LOG 
SAVE_SAMPLES_STEP = 200 # Generate and save samples every SAVE_SAMPLES_STEP
CHECKPOINT_STEP = 4000

OUTPUT_DIM = N_PIXELS * N_PIXELS * N_COLORS # Number of pixels in each iamge
N_GPUS = 1 # Number of GPUs

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
if DATASET == 'lsun':
    Generator, Discriminator = network.gan64_model.GeneratorAndDiscriminator()
elif DATASET == 'mnist' or DATASET == 'svhn':
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
            fake_labels_splits = real_labels
            fake_data = Generator('New.', BATCH_SIZE//len(DEVICES), labels = fake_labels_splits)
            fake_data_old = Generator('Old.', BATCH_SIZE//len(DEVICES), labels = fake_labels_splits)

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

    _fixed_noise = pickle.load(open('/datatmp/result/fixed_noise/fixed_noise','r'))
    fixed_noise_one = tf.constant(_fixed_noise)
    fixed_noise_samples = []
    for i in range(NUM_CLASS):
        fixed_labels_one = tf.constant(np.ones(100,dtype='int32')*i)
        fixed_noise_samples.append(Generator('New.', 100, labels = fixed_labels_one, noise=fixed_noise_one))

    def generate_image_one(num_classes, iterations):
        for i in range(num_classes):
            samples = session.run(fixed_noise_samples[i])
            samples = ((samples+1.)*(255./2)).astype('int32')
            samples = samples.reshape((100, N_COLORS, N_PIXELS, N_PIXELS))
            if DATASET == 'mnist':
                samples = np.tile(samples,[1,3,1,1])
            lib.save_images.save_images(samples, '{}Task_{}_Iter_{}_{}.png'.format(SAMPLES_DIR, num_classes, iterations, all_classes[i]))

    ckpt_saver = tf.train.Saver(max_to_keep=10000)
    session.run(tf.global_variables_initializer())

    if args.test:
        if DATASET == 'svhn' or DATASET == 'mnist':
            labelmap = list(range(10))
        elif DATASET == 'lsun':
            labelmap = [0,1,8,6]

        model = pretrain_classifier(session, image_size = N_PIXELS, batch_size = BATCH_SIZE, dataset = DATASET)
        if DATASET == 'mnist' or DATASET == 'svhn':
            task_test = [4,9]           # Evaluate models after task 4 (5th) and task9 (10th)
        else:
            task_test = [0,1,2,3]       # Evaluate models after each task

        for task in task_test:
            LOAD_MODEL_FILE = MODEL_DIR + 'WGAN_GP.model' + str(all_classes[task]) + '-' + str(ITERS - 1)
            ckpt_saver.restore(session, LOAD_MODEL_FILE)
            ave_acc = 0
            for label in range(task+1):
                samples = Generator('New.', BATCH_SIZE, labels = tf.constant(np.ones(BATCH_SIZE)*label,dtype='int32'))
                acc = 0.
                repeat = 100
                for i in range(repeat):
                    _samples = session.run(samples) * 128 + 128
                    _samples = _samples.astype(int).reshape([BATCH_SIZE,N_COLORS,N_PIXELS,N_PIXELS]).transpose([0,2,3,1]).squeeze()
                    predicted_labels, predicted_probality = model.test(_samples)
                    _acc = float(np.sum(np.equal(predicted_labels,labelmap[label]))) / BATCH_SIZE
                    acc += _acc/repeat
                print('Task: {}   Class: {}   Accuracy: {}'.format(task + 1, all_classes[label], acc))
                ave_acc += acc / (task+1)
            print('Average accuracy: {}'.format(ave_acc))
        exit()

    class_count = 0
    if LOAD_MODEL:
        ckpt_saver.restore(session, LOAD_MODEL_FILE)
        for var_name in lib._params:
            if 'New.Generator' in var_name:
                name = var_name[4:]
                value = session.run(lib._params['New.' + name])
                session.run(lib._params['Old.' + name].assign(value))
        class_count = 1

    # Train loop
    for classes in all_classes[class_count:]:
        _labels = np.ones(BATCH_SIZE).astype(int) * class_count

        # Dataset for current task
        if DATASET == 'lsun':
            train_gen, dev_gen = lib.lsun.load(BATCH_SIZE, [classes], data_dir = DATASET_DIR)
        elif DATASET == 'mnist':
            train_gen, dev_gen, _ = lib.mnist.load(BATCH_SIZE, [classes], data_dir = DATASET_DIR)
        elif DATASET == 'svhn':
            train_gen, dev_gen = lib.svhn.load(BATCH_SIZE, [classes], data_dir = DATASET_DIR)
        def inf_train_gen():
            while True:
                for (images,labels) in train_gen():
                    yield images,labels
        gen = inf_train_gen()

        # Save a batch of ground-truth samples
        _x,_ = inf_train_gen().next()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
        _x_r = ((_x_r+1.)*(255.99//2)).astype('int32')
        if DATASET == 'lsun' or DATASET == 'svhn':
            lib.save_images.save_images(_x_r.reshape((BATCH_SIZE//N_GPUS, N_COLORS, N_PIXELS, N_PIXELS)), '%s/samples_groundtruth_%s.png' % (SAMPLES_DIR, classes))
        elif DATASET == 'mnist':
            lib.save_images.save_images(np.tile(_x_r.reshape([BATCH_SIZE//N_GPUS,N_COLORS,N_PIXELS,N_PIXELS]),[1,3,1,1]), '%s/samples_groundtruth_%s.png' % (SAMPLES_DIR, classes))

        # Loss function for keeping previous tasks
        if class_count > 0:
            if RA:      # MeRGAN Replay Alignment
                RA_costs = []
                def RA_loss_fn(fake_labels,num_samples):
                    noise = tf.random_normal([num_samples, 128])
                    old_imgs = Generator('Old.', num_samples, labels = fake_labels, noise = noise)
                    new_imgs = Generator('New.', num_samples, labels = fake_labels, noise = noise)                
                    return tf.nn.l2_loss(old_imgs - new_imgs)
                num_samples = n_samples // class_count if REDUCE_BATCH else n_samples
                for i in range(class_count):
                    fake_labels = tf.ones(num_samples, tf.int32) * i
                    RA_costs.append(RA_loss_fn(fake_labels, num_samples))
                RA_cost = tf.add_n(RA_costs) * RA_FACTOR
                RA_cost = RA_cost / class_count if REDUCE_BATCH else RA_cost
            else:
                RA_cost = tf.constant(0.)
            
            if JTR:     # MeRGAN Joint Training with Replay
                JTR_gen_costs, JTR_gen_acgan_costs, JTR_disc_costs, JTR_disc_acgan_costs, JTR_old_accs, JTR_new_accs = [],[],[],[],[],[]

                def JTR_loss_fn(fake_labels,num_samples):                    
                    noise = tf.random_normal([num_samples, 128])

                    fake_data_old = Generator('Old.', num_samples, labels = fake_labels, noise = noise)
                    fake_data_new = Generator('New.', num_samples, labels = fake_labels, noise = noise)

                    disc_JTR_old, disc_JTR_old_acgan = Discriminator(fake_data_old)
                    disc_JTR_new, disc_JTR_new_acgan = Discriminator(fake_data_new)
                    JTR_disc_costs.append(tf.reduce_mean(disc_JTR_new) - tf.reduce_mean(disc_JTR_old))
                    JTR_gen_costs.append(-tf.reduce_mean(disc_JTR_new))

                    JTR_disc_acgan_costs.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_JTR_old_acgan, labels=fake_labels)))
                    JTR_gen_acgan_costs.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_JTR_new_acgan, labels=fake_labels)))

                    JTR_old_accs.append(tf.reduce_mean(
                        tf.cast(tf.equal(tf.to_int32(tf.argmax(disc_JTR_old_acgan, dimension=1)), fake_labels ), tf.float32)))
                    JTR_new_accs.append(tf.reduce_mean(
                        tf.cast(tf.equal(tf.to_int32(tf.argmax(disc_JTR_new_acgan, dimension=1)), fake_labels ), tf.float32)))
                
                num_samples = n_samples // class_count if REDUCE_BATCH else n_samples
                for i in range(class_count):
                    fake_labels = tf.ones(num_samples, tf.int32) * i
                    JTR_loss_fn(fake_labels,num_samples)
                n_divide = class_count

                JTR_disc_cost = (tf.add_n(JTR_disc_costs) + tf.add_n(JTR_disc_acgan_costs))
                JTR_gen_cost = (tf.add_n(JTR_gen_costs) + tf.add_n(JTR_gen_acgan_costs))
                JTR_old_acc = tf.add_n(JTR_old_accs) / n_divide
                JTR_new_acc = tf.add_n(JTR_new_accs) / n_divide

                JTR_disc_cost = JTR_disc_cost if REDUCE_BATCH else JTR_disc_cost / n_divide
                JTR_gen_cost = JTR_gen_cost if REDUCE_BATCH else JTR_gen_cost / n_divide
  
            else:
                JTR_disc_cost = tf.constant(0.)
                JTR_gen_cost = tf.constant(0.)

            disc_cost_all = disc_cost + JTR_disc_cost
            gen_cost_all = gen_cost + RA_cost + JTR_gen_cost


            gen_train_op = tf.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost_all,
                                           var_list=lib.params_with_name('New.Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9).minimize(disc_cost_all,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
    
        else:
            gen_train_op = tf.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost,
                                       var_list=lib.params_with_name('New.Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        session.run(tf.variables_initializer([var for var in tf.global_variables() if 'Adam' in var.name]))
        session.run(tf.variables_initializer([var for var in tf.global_variables() if 'beta' in var.name]))

        # Train
        for iteration in range(ITERS):
            start_time = time.time()

            # Train generator
            if iteration > 0:    
                if class_count > 0:
                    _gen_cost, _JTR_disc_cost, _JTR_gen_cost, _RA_cost, _ = session.run([gen_cost,  
                                JTR_disc_cost, JTR_gen_cost, RA_cost, gen_train_op], 
                                                            feed_dict = {all_real_labels: _labels})
                else:
                    _gen_cost, _ = session.run([gen_cost, gen_train_op], 
                                                            feed_dict = {all_real_labels: _labels})
                lib.plot.plot('g-cost', _gen_cost)
                if RA and class_count > 0:
                    lib.plot.plot('RA-cost', _RA_cost)
                if JTR and class_count > 0:
                    lib.plot.plot('JTR_gen_cost', _JTR_gen_cost)
                    lib.plot.plot('JTR_disc_cost', _JTR_disc_cost)

            # Train critic
            for i in range(CRITIC_ITERS):
                _images, _ = gen.next()
                _disc_cost, _disc_wgan, _disc_wgan_pure,  _ = session.run(
                    [disc_cost, disc_wgan, disc_wgan_pure,  disc_train_op], 
                        feed_dict={all_real_data_conv: _images, all_real_labels: _labels})

            lib.plot.plot('d-cost', _disc_cost)
            lib.plot.plot('wgan-pure', _disc_wgan_pure)
            lib.plot.plot('penalty', _disc_wgan - _disc_wgan_pure)
            lib.plot.plot('time', time.time() - start_time)

            # Generate samples
            if iteration % SAVE_SAMPLES_STEP == 0 or iteration in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]:                
                generate_image_one(class_count + 1, iteration)

            # Save checkpoint
            if iteration % CHECKPOINT_STEP == 0:
                ckpt_saver.save(session, MODEL_DIR + 'WGAN_GP.model' + str(classes), iteration)
            if iteration < 10 or iteration % 100 == 0:
                lib.plot.flush(path = RESULT_DIR)
            lib.plot.tick()

        # Copy Generator
        class_count += 1
        for var_name in lib._params:
            if 'New.Generator' in var_name:
                name = var_name[4:]
                value = session.run(lib._params['New.' + name])
                session.run(lib._params['Old.' + name].assign(value))