import tensorflow as tf
import sys
sys.path.append('../')
from AbstractEncoder import *

class Encoder(AbstractEncoder):
    # Override the layers construction function
    def constructLayers(self, input):
        conv1 = tf.layers.conv2d(input, filters=32, kernel_size=(3,3), padding='same', \
                                 activation=tf.nn.relu)
        # Now 28x28x32
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 14x14x32
        conv2 = tf.layers.conv2d(maxpool1, filters=32, kernel_size=(3,3), padding='same', \
                                 activation=tf.nn.relu)
        # Now 14x14x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 7x7x32
        conv3 = tf.layers.conv2d(maxpool2, filters=16, kernel_size=(3,3), padding='same', \
                                 activation=tf.nn.relu)
        # Now 7x7x16
        encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
        # Now 4x4x16

        return encoded
