import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class SimpleQuadratic(Layer):

    def __init__(self, units=32, activation=None):
        """ Initializes the class and sets up the internal variables """
        super(SimpleQuadratic, self).__init__()
        self.units = units

        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """ Create the state of the layer (weights) """
        # a and b should be initialized with random normal, c (or the bias) with zeros.
        # remember to set these as trainable.
        # YOUR CODE HERE
        a_init = tf.random_normal_initializer()
        self.a = tf.Variable(name="a", initial_value=a_init(shape=(input_shape[-1], self.units), dtype="float32"),
                             trainable=True)
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(name="b", initial_value=b_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True)
        c_init = tf.zeros_initializer()
        self.c = tf.Variable(name="c", initial_value=c_init(shape=(self.units,), dtype='float32'),
                             trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """ Defines the computation from inputs to outputs """
        return self.activation(tf.matmul(K.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c)


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        SimpleQuadratic(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)