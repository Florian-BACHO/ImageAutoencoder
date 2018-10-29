import tensorflow as tf

class AbstractEncoder:
    def __init__(self, input):
        self.input = input

        self.output = self.constructLayers(input)

    # To override
    def constructLayers(self, input):
        raise NotImplementedError

    def __call__(self, entry):
        session = tf.get_default_session()

        return session.run(self.output, feed_dict={self.input: entry})
