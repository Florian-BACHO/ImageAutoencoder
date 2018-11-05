import tensorflow as tf
import numpy as np
from Encoder import *
from Decoder import *

BATCH_SIZE = 32
LEARNING_RATE = 0.01

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add dimension to fit with shape: (None, Height, Width, 1)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    height = x_train.shape[1]
    width = x_train.shape[2]

    input = tf.placeholder(tf.float32, (None, height, width, 1))

    with tf.variable_scope("Encoder"):
        encoder = Encoder(input)

    with tf.variable_scope("Decoder"):
        decoder = Decoder(input, encoder, learning_rate=LEARNING_RATE)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            batch = x_train[np.random.choice(x_train.shape[0], BATCH_SIZE)]

            loss_train = decoder.train(batch)

            if i % 10 == 0:
                batch = x_test[np.random.choice(x_test.shape[0], BATCH_SIZE)]
                loss_test = decoder.evaluate(batch)
                print("Epoch %d: Training Loss = %f, Test Loss = %f" % (i, loss_train, loss_test))
