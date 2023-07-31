import keras
import tensorflow as tf
import numpy as np

from keras.layers import *
from keras.initializers.initializers_v2 import HeNormal, HeUniform

from random import seed, randint, random


class TemperaturePredictor(tf.keras.Model):

    def __init__(self, custom_seed, output_dim):
        super(TemperaturePredictor, self).__init__()

        seed(custom_seed)
        ran_min = round(random() * 10)
        ran_max = round(random() * (10 ** 6))

        self.modules = []
        self.modules.append(Dense(35, activation='relu',
                  kernel_initializer=HeNormal(seed=randint(ran_min, ran_max)),
                  bias_initializer=tf.keras.initializers.Constant(value=0.01)))
        for i in range(1, 3):
            self.modules.append(Dense(35, activation='relu',
                                 kernel_initializer=HeNormal(seed=randint(ran_min, ran_max)),
                                 bias_initializer=tf.keras.initializers.Constant(value=0.01)))

        self.modules.append(Dense(output_dim, activation='linear',
                             kernel_initializer=HeUniform(seed=randint(ran_min, ran_max)),
                             bias_initializer=tf.keras.initializers.Constant(value=0)))

        self.model = keras.Sequential(self.modules)

    def select_component(self, x, condition):
        L = x.shape[-1]
        if condition == 0:
            if len(x.shape) == 2:
                return x[:, :L // 2]
            return x[:, :, :, :L // 2]
        else:
            if len(x.shape) == 2:
                return x[:, L // 2:]
            return x[:, :, :, L // 2:]

    def call(self, input, minmax):
        x = self.model.call(input)
        x = self.select_component(x, minmax)
        # x = tf.squeeze(x)
        # print("component{}".format(minmax), x)
        return x


class TemperaturePredictor2(tf.keras.Model):

    def __init__(self, custom_seed, output_dim):
        super(TemperaturePredictor2, self).__init__()

        self.noise = tf.Variable(tf.cast(np.log(10.), tf.float32), trainable=True, name="Variable")

        seed(custom_seed)
        self.modules = []
        self.modules.append(Dense(35, activation='relu'))
        for i in range(1, 3):
            self.modules.append(Dense(35, activation='relu'))
            self.modules.append(Dense(output_dim, activation='sigmoid'))

        self.model = keras.Sequential(self.modules)

    def call(self, input):
        x = self.model.call(input)
        x = x * (19.3 + 9.7) - 9.7
        noise = tf.ones_like(x) * self.noise
        # x = tf.stack([x, tf.math.exp(noise)], axis=-1)[:, 0, :]
        return [x, tf.math.exp(noise)]