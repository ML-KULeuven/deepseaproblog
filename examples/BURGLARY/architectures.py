import tensorflow as tf
import math

from keras.layers import *


class MNIST_CNN(tf.keras.Model):

    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = Conv2D(6, 5)
        self.conv2 = Conv2D(16, 5)
        self.max = MaxPool2D()
        self.relu = ReLU()
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max(x)
        x = self.relu(x)
        return self.relu(self.max(self.conv2(x)))


class MNIST_Classifier(tf.keras.Model):

    def __init__(self, conv, N=10, with_softmax=True):
        super(MNIST_Classifier, self).__init__()
        self.conv = conv
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = Softmax()
        self.classifier = tf.keras.Sequential()
        self.classifier.add(Conv2D(N, 2))
        self.classifier.add(MaxPool2D())
        self.classifier.add(ReLU())

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.classifier(x)
        x = tf.reduce_mean(x, axis=(1, 2))
        if self.with_softmax:
            x = self.softmax(x)
        return x


class Parameter_net(tf.keras.Model):

    def __init__(self, mean_var, var_var):
        super(Parameter_net, self).__init__()
        self.mean = tf.Variable(mean_var, name="mean", trainable=True)
        self.sigma = tf.Variable(math.log(var_var), name="sigma", trainable=True)


    def call(self, x):
        return [self.mean, tf.exp(self.sigma)]


class JointParameterNet(tf.keras.Model):

    def __init__(self, means, variances):
        super(JointParameterNet, self).__init__()
        self.means = tf.Variable(means)
        self.variances = tf.Variable(tf.math.log(variances))
    
    def call(self, x):
        return [self.means, tf.exp(self.variances)]


class MNIST_Net(tf.keras.Model):

    def __init__(self, encoder, N=10):
        super(MNIST_Net, self).__init__()
        self.encoder = encoder
        self.flatten = Flatten()
        self.classifier = tf.keras.Sequential()
        self.classifier.add(Dense(120))
        self.classifier.add(ReLU())
        self.classifier.add(Dense(84))
        self.classifier.add(ReLU())
        self.classifier.add(Dense(N))
        self.softmax = Softmax()

    def call(self, input):
        x = self.encoder(input)
        x = self.flatten(x)
        x = self.classifier(x)
        return self.softmax(x)
