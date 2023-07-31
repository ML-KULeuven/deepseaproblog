import tensorflow as tf

from keras.layers import *


class DigitClassifier(tf.keras.Model):

    def __init__(self, latent_dim, N=10):
        super(DigitClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.N = N
        self.eval = False

        modules = []
        modules.append(Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
        modules.append(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
        modules.append(Flatten())
        modules.append(Dense(latent_dim))
        modules.append(Dense(32))
        modules.append(ReLU())
        modules.append(Dense(24))
        modules.append(ReLU())
        modules.append(Dense(N))
        modules.append(Softmax())

        self.model = tf.keras.Sequential(modules)

    def call(self, input):
        x = self.model.call(input)
        return x


class DenseEncoder(tf.keras.Model):

    def __init__(self, latent_dim):
        super(DenseEncoder, self).__init__()
        self.eval = False

        self.flatten = Flatten()
        self.dense = Dense(200, activation='relu')
        self.dense2 = Dense(100, activation='relu')
        self.mean = Dense(latent_dim, activation='linear')
        self.sd = Dense(latent_dim, activation='linear')

    def call(self, input):
        x = self.flatten(input)
        x = self.dense(x)
        x = self.dense2(x)
        mean = self.mean(x)
        if self.eval:
            sd = tf.zeros_like(mean)
        else:
            sd = tf.exp(0.5 * self.sd(x))
        return [mean, sd]


class DenseDecoder(tf.keras.Model):

    def __init__(self):
        super(DenseDecoder, self).__init__()
        self.dense = Dense(200, activation='relu')
        self.dense2 = Dense(100, activation='relu')
        self.out = Dense(784, activation='tanh')

    def call(self, latent, condition):
        if condition.dtype != tf.float32:
            condition2 = tf.one_hot(condition, depth=10)
            condition2 = tf.reshape(condition2, [-1, 10])
            condition2 = tf.repeat(condition2, repeats=latent.shape[0], axis=0)
            x = tf.concat([latent[:, :], condition2], axis=-1)
        else:
            x = tf.concat([latent, condition], axis=-1)
        x = self.dense(x)
        x = self.dense2(x)
        x = self.out(x)
        return tf.reshape(x, [-1, 28, 28, 1])


class Encoder(tf.keras.Model):

    def __init__(self, latent_dim=2, simple=True):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.eval = False

        modules = []
        modules.append(Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
        if not simple:
            modules.append(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
        modules.append(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
        modules.append(Flatten())

        self.encoder = tf.keras.Sequential(modules)
        self.mean = Dense(latent_dim)
        self.sd = Dense(latent_dim)

    def call(self, input):
        x = self.encoder(input)
        mean = self.mean(x)
        if self.eval:
            sd = tf.zeros_like(mean)
        else:
            sd = tf.exp(0.5 * self.sd(x))
        return [mean, sd]


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=2, simple=True):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if simple:
            modules.append(Dense(7 * 7 * 32, activation=tf.nn.relu))
            modules.append(Reshape(target_shape=(7, 7, 32)))
        else:
            modules.append(Dense(8 * 8 * 64, activation=tf.nn.relu))
            modules.append(Reshape(target_shape=(8, 8, 64)))
        modules.append(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'))
        if not simple:
            modules.append(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'))
        modules.append(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'))
        modules.append(Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='tanh'))
        self.decoder = tf.keras.Sequential(modules)

    def call(self, latent, condition):
        if condition.dtype != tf.float32:
            condition2 = tf.one_hot(condition, depth=10)
            condition2 = tf.reshape(condition2, [-1, 10])
            condition2 = tf.repeat(condition2, repeats=latent.shape[0], axis=0)
            x = tf.concat([latent[:, :], condition2], axis=-1)
        else:
            x = tf.concat([latent, condition], axis=-1)
        out = self.decoder(x)
        return out
