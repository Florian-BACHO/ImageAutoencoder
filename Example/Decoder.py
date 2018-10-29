import tensorflow as tf
import sys
sys.path.append('../')
from AbstractDecoder import *

class Decoder(AbstractDecoder):
    # Override the layers construction function
    def constructLayers(self, input):
        upsample1 = tf.image.resize_images(input, size=(7,7), \
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 7x7x16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), \
                                 padding='same', activation=tf.nn.relu)
        # Now 7x7x16
        upsample2 = tf.image.resize_images(conv4, size=(14,14), \
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 14x14x16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), \
                                 padding='same', activation=tf.nn.relu)
        # Now 14x14x32
        upsample3 = tf.image.resize_images(conv5, size=(28,28), \
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Now 28x28x32
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), \
                                 padding='same', activation=tf.nn.relu)
        # Now 28x28x32

        logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), \
                                  padding='same', activation=None)
        #Now 28x28x1
        return logits
