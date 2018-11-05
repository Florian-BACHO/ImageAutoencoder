import tensorflow as tf

class AbstractEncoder:
    def __init__(self, input):
        self.input = input
        self.isTraining = tf.placeholder(tf.bool, ())

        self.output = self.constructLayers(input, self.isTraining)

    # To override
    # isTraining is a boolean tensor that can be used for batch normalization
    def constructLayers(self, input):
        raise NotImplementedError

    def __call__(self, entry):
        session = tf.get_default_session()

        return session.run(self.output, feed_dict={self.input: entry, \
                                                   self.isTraining: False})
