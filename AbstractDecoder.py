import tensorflow as tf

class AbstractDecoder:
    def __init__(self, input, encoder, learning_rate=0.01):
        self.input = input

        layers = self.constructLayers(encoder.output)
        self.output = tf.nn.sigmoid(layers)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=layers)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(loss)

    def constructLayers(self, input):
        raise NotImplementedError

    def __call__(self, entry):
        session = tf.get_default_session()

        return session.run(self.output, feed_dict={self.input: entry})

    def train(self, entry):
        session = tf.get_default_session()

        loss_v, _ = session.run([self.loss, self.training_op], feed_dict={self.input: entry})
        return loss_v

    def evaluate(self, entry):

        session = tf.get_default_session()

        loss_v = session.run(self.loss, feed_dict={self.input: entry})
        return loss_v
