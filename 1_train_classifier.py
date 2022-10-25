import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import functools
from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession, app, gfile, ConfigProto, placeholder
tf.compat.v1.disable_v2_behavior()
from pprint import pprint

import layers.optics as optics
from layers.utils import fft_conv2d

CONFIG = type('', (), {})() # object for params
CONFIG.dropout = 0.9
CONFIG.batch_size = 128
CONFIG.num_iters = 10001
CONFIG.learning_rate = 5e-4
CONFIG.learning_rate_ad = 1
CONFIG.isNonNeg = True
CONFIG.opt_type = 'ADAM'
CONFIG.pad_amount = 64
CONFIG.tiling_factor = 4
CONFIG.kernel_size = 32
CONFIG.tile_size = 40
CONFIG.classes = 16
CONFIG.summary_every = 10
CONFIG.print_every = 10
CONFIG.save_every = 1000
CONFIG.verbose = True
CONFIG.train_data_path = './assets/quickdraw16_train.npy'
CONFIG.test_data_path = './assets/quickdraw16_test.npy'

now = datetime.now()
runtime = now.strftime('%Y%m%d-%H%M%S')
run_id = 'quickdraw_tiled_nonneg/' + runtime + '/'
CONFIG.log_dir = os.path.join('checkpoints/', run_id)

print()
pprint(CONFIG.__dict__)
print()

if gfile.Exists(CONFIG.log_dir):
    gfile.DeleteRecursively(CONFIG.log_dir)
gfile.MakeDirs(CONFIG.log_dir)

sess = InteractiveSession(config=ConfigProto(allow_soft_placement=True))

# MODEL
# input placeholders
with tf.compat.v1.name_scope('input'):
    inputs = placeholder(tf.float32, shape=[None, 784])
    target = placeholder(tf.int64, shape=[None, CONFIG.classes])
    keep_prob = placeholder(tf.float32)

    x_image = tf.reshape(inputs, [-1, 28, 28, 1]) 
    paddings = tf.constant(
        [[0, 0,], 
         [CONFIG.pad_amount, CONFIG.pad_amount], # height 
         [CONFIG.pad_amount, CONFIG.pad_amount], # width
         [0, 0]]
    )
    x_image = tf.pad(tensor = x_image, paddings = paddings)
    tf.compat.v1.summary.image('input', x_image, 3)
    print("\nx_image shape: ", x_image.get_shape().as_list(), end='\n\n')

global_step = tf.Variable(0, trainable=False)

def tiled_conv_layer(
        input_img, 
        tiling_factor, 
        tile_size, 
        kernel_size,
        name = 'tiling_conv', 
        regularizer = None, 
        nonneg = False
    ):
    dims = input_img.get_shape().as_list()

    with tf.compat.v1.variable_scope(name):
        kernel_lists = []
        for i in range(tiling_factor):
            tmp = []
            for j in range(tiling_factor):
                initializer = tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, 
                    mode="fan_avg", 
                    distribution="uniform"
                )
                tmp.append(
                    tf.compat.v1.get_variable(
                        'kernel_%d%d'%(i,j),
                        shape = (kernel_size, kernel_size, 1, 1),
                        initializer = initializer
                    )
                )
            kernel_lists.append(tmp)
        
        # maybe pad_before = pad_after?
        pad_before = np.ceil((tile_size - kernel_size)/2).astype(np.uint32)
        pad_after = np.floor((tile_size - kernel_size)/2).astype(np.uint32)

        padded_kernels = []
        for kernels in kernel_lists:
            tmp = []
            for kernel in kernels:
                tmp.append(
                    tf.pad(
                        tensor=kernel, 
                        paddings=[[pad_before, pad_after], [pad_before, pad_after], [0,0], [0,0]]
                    )
                )
            padded_kernels.append(tmp)
        
        psf = tf.concat([tf.concat(kernel_list, axis=0) for kernel_list in padded_kernels], axis=1)
        
        if nonneg:
            psf = tf.abs(psf)
        
        output_img = fft_conv2d(input_img, psf)
        
        return output_img

# single tiled convolutional layer
h_conv1 = tiled_conv_layer(
    input_img = x_image, 
    tiling_factor = CONFIG.tiling_factor, 
    tile_size = CONFIG.tile_size,
    kernel_size = CONFIG.kernel_size,
    name = 'h_conv1',
    nonneg = CONFIG.isNonNeg
)

split_1d = tf.split(h_conv1, num_or_size_splits=4, axis=1)
# calculating output scores
h_conv_split = tf.concat(
    [tf.split(split_1d[0], num_or_size_splits=4, axis=2),
     tf.split(split_1d[1], num_or_size_splits=4, axis=2),
     tf.split(split_1d[2], num_or_size_splits=4, axis=2),
     tf.split(split_1d[3], num_or_size_splits=4, axis=2)], 
    axis=0
)
predicted = tf.transpose(a=tf.reduce_max(input_tensor=h_conv_split, axis=[2,3,4]))

tf.compat.v1.summary.image('output', tf.reshape(predicted, [-1, 4, 4, 1]), 3)

# loss, train, acc
with tf.compat.v1.name_scope('cross_entropy'):
    total_data_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target), logits=predicted)
    data_loss = tf.reduce_mean(input_tensor=total_data_loss)
    reg_loss = tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = tf.add(data_loss, reg_loss)
    tf.compat.v1.summary.scalar('data_loss', data_loss)
    tf.compat.v1.summary.scalar('reg_loss', reg_loss)
    tf.compat.v1.summary.scalar('total_loss', total_loss)

if CONFIG.opt_type == 'ADAM':
    train_step = tf.compat.v1.train.AdamOptimizer(CONFIG.learning_rate).minimize(total_loss, global_step)
elif CONFIG.opt_type == 'Adadelta':
    train_step = tf.compat.v1.train.AdadeltaOptimizer(CONFIG.learning_rate_ad, rho=.9).minimize(total_loss, global_step)
else:
    train_step = tf.compat.v1.train.MomentumOptimizer(CONFIG.learning_rate, momentum=0.5, use_nesterov=True).minimize(total_loss, global_step)

with tf.compat.v1.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(input=predicted, axis=1), tf.argmax(input=target, axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', accuracy)

losses = []

# tensorboard setup
merged = tf.compat.v1.summary.merge_all()
train_writer = tf.compat.v1.summary.FileWriter(CONFIG.log_dir + '/train', sess.graph)
test_writer = tf.compat.v1.summary.FileWriter(CONFIG.log_dir + '/test')

tf.compat.v1.global_variables_initializer().run()

# add ops to save and restore all the variables
saver = tf.compat.v1.train.Saver(max_to_keep=2)
save_path = os.path.join(CONFIG.log_dir, 'model.ckpt')

# change to your directory
train_data = np.load(CONFIG.train_data_path)
test_data = np.load(CONFIG.test_data_path)

def get_feed(train, batch_size):
    if train:
        idcs = np.random.randint(0, np.shape(train_data)[0], batch_size)
        x = train_data[idcs, :]
        y = np.zeros((batch_size, CONFIG.classes))
        y[np.arange(batch_size), idcs//8000] = 1
        
    else:
        x = test_data
        y = np.zeros((np.shape(test_data)[0], CONFIG.classes))
        y[np.arange(np.shape(test_data)[0]), np.arange(np.shape(test_data)[0])//100] = 1                
    
    return x, y

x_test, y_test = get_feed(train=False, batch_size=CONFIG.batch_size)

# TRAIN LOOP
for i in range(CONFIG.num_iters):
    x_train, y_train = get_feed(train=True, batch_size=CONFIG.batch_size)
    _, loss, reg_loss_graph, train_accuracy, train_summary = sess.run(
        [train_step, total_loss, reg_loss, accuracy, merged], 
        feed_dict={inputs: x_train, target: y_train, keep_prob: CONFIG.dropout}
    )
    losses.append(loss)

    if i % CONFIG.summary_every == 0:
        train_writer.add_summary(train_summary, i)
        
    if i > 0 and i % CONFIG.save_every == 0:
        # print("Saving model...")
        saver.save(sess, save_path, global_step=i)
        
    if i % CONFIG.print_every == 0:
        if CONFIG.verbose:
            print('\rstep %d:\t loss %g,\t reg_loss %g,\t train acc %g' %
                    (i, loss, reg_loss_graph, train_accuracy), end="")
            

test_batches = []
# TEST LOOP
for i in range(32):
    idx = i*50
    batch_acc = accuracy.eval(feed_dict={inputs: x_test[idx:idx+50, :], target: y_test[idx:idx+50, :], keep_prob: 1.0})
    test_batches.append(batch_acc)
test_acc = np.mean(test_batches)   

#test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
print('final step %d, train accuracy %g, test accuracy %g' %
        (i, train_accuracy, test_acc))
sess.close()

train_writer.close()
test_writer.close()