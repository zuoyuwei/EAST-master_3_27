#author:zuo
#data:2020_3_27
#usage:use vgg16 replace that resnet_v1_50 model
import tensorflow as tf

def vgg16_network(input):
    net = tf.layers.conv2d(input, 64, 3, (1, 1), padding='SAME', name='conv1_1')
    net = tf.layers.conv2d(net, 64, 3, (1, 1), padding='SAME', name='conv1_2')
    net = tf.layers.max_pooling2d(net, 2, 2, padding='VALID', name='pool1')
    net = tf.layers.conv2d(net, 128, 3, (1, 1), padding='SAME', name='conv2_1')
    net = tf.layers.conv2d(net, 128, 3, (1, 1), padding='SAME', name='conv2_2')
    net = tf.layers.max_pooling2d(net, 2, 2, padding='VALID', name='pool2')
    net = tf.layers.conv2d(net, 256, 3, (1, 1), padding='SAME', name='conv3_1')
    net = tf.layers.conv2d(net, 256, 3, (1, 1), padding='SAME', name='conv3_2')
    net = tf.layers.conv2d(net, 256, 3, (1, 1), padding='SAME', name='conv3_3')
    net = tf.layers.max_pooling2d(net, 2, 2, padding='VALID', name='pool3')
    net = tf.layers.conv2d(net, 512, 3, (1, 1), padding='SAME', name='conv4_1')
    net = tf.layers.conv2d(net, 256, 3, (1, 1), padding='SAME', name='conv4_2')
    net = tf.layers.conv2d(net, 256, 3, (1, 1), padding='SAME', name='conv4_3')
    net = tf.layers.max_pooling2d(net, 2, 2, padding='VALID', name='pool4')

