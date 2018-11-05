import tensorflow as tf

class AbstractDecoder:
    def __init__(self, input, encoder, learning_rate=0.01):
        self.input = input
        self.isTraining = encoder.isTraining

        self.output = self.constructLayers(encoder.output, self.isTraining)

        loss = tf.losses.mean_squared_error(labels=input, predictions=self.output)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(loss)

    # To override
    # isTraining is a boolean tensor that can be used for batch normalization
    def constructLayers(self, input, isTraining):
        raise NotImplementedError

    def __call__(self, entry):
        session = tf.get_default_session()

        return session.run(self.output, feed_dict={self.input: entry, \
                                                   self.isTraining: False})

    def train(self, entry):
        session = tf.get_default_session()

        loss_v, _ = session.run([self.loss, self.training_op], feed_dict={self.input: entry, \
                                                                          self.isTraining: True})
        return loss_v

    def evaluate(self, entry):

        session = tf.get_default_session()

        loss_v = session.run(self.loss, feed_dict={self.input: entry, \
                                                   self.isTraining: False})
        return loss_v
