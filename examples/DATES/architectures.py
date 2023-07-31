import tensorflow as tf
import tensorflow_probability as tfp

from keras.layers import *


class HorizontalDifference(tf.keras.Model):

    def __init__(self):
        super().__init__()

    def call(self, x, y):
        if len(x.shape) == 0:
            return tf.expand_dims(x - y[:, :, 0, :], axis=-2)
        return tf.expand_dims(x[:, :, 0, :] - y[:, :, 0, :], axis=-2)

class AlexConvolutions(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.modules = []
        # self.modules.append(Conv2D(16, (7, 11), strides=4, activation=ReLU()))
        # self.modules.append(MaxPool2D((2, 3), strides=2))
        # self.modules.append(Conv2D(32, (3, 5), activation=ReLU()))
        # self.modules.append(MaxPool2D((2, 3), strides=2))
        # self.modules.append(Conv2D(48, (2, 3), activation=ReLU()))
        # self.modules.append(Conv2D(48, (2, 3), activation=ReLU()))
        # self.modules.append(Conv2D(32, (2, 3), activation=ReLU()))
        # self.modules.append(MaxPool2D((2, 3), strides=2))
        self.modules.append(Conv2D(16, 5, activation=ReLU()))
        self.modules.append(MaxPool2D())
        self.modules.append(Conv2D(32, 5, activation=ReLU()))
        self.modules.append(MaxPool2D())
        self.modules.append(Conv2D(64, 5, activation=ReLU()))
        self.modules.append(MaxPool2D())
        # self.modules.append(Conv2D(128, 5, activation=ReLU()))
        # self.modules.append(MaxPool2D())

        self.model = tf.keras.Sequential(self.modules)

    def call(self, x, **kwargs):
        return self.model.call(x)


class AlexRegressor(tf.keras.Model):

    def __init__(self, backbone, size=128, blocks=4):
        super().__init__()
        self.backbone = backbone
        self.blocks = blocks
        self.size = size

        self.modules = [self.backbone]
        self.modules.append(Flatten())
        self.modules.append(Dense(size, activation=ReLU()))
        self.modules.append(Dense(size // 2, activation=ReLU()))
        self.modules.append(Dense(blocks * 4))

        self.model = tf.keras.Sequential(self.modules)

    def call(self, x, **kwargs):
        """
        out: list of parameters of 2 dimensional generalised normal. One for each block.
        """
        x = self.model.call(x)
        mu_x, mu_y, sigma_x, sigma_y = x[:, :self.blocks], x[:, self.blocks:2*self.blocks], \
            x[:, 2*self.blocks:3*self.blocks], x[:, 3*self.blocks:4*self.blocks]
        # mu_x, mu_y = x[:, :self.blocks], x[:, self.blocks:2*self.blocks]
        mu_x, mu_y = (tf.math.tanh(mu_x) + 1) / 2, (tf.math.tanh(mu_y) + 1) / 2
        # mu_x, mu_y = tf.math.sigmoid(mu_x), tf.math.sigmoid(mu_y)
        # sigma_x, sigma_y = 0.01 + tf.exp(-sigma_x), 0.01 + tf.exp(-sigma_y)
        # sigma_x, sigma_y = 0.01 + tf.math.sigmoid(sigma_x), 0.01 + tf.math.sigmoid(sigma_y)
        # sigma_x, sigma_y = (tf.math.tanh(sigma_x) + 1) / 2, (tf.math.tanh(sigma_y) + 1) / 2
        sigma_x, sigma_y = 0.01 + (tf.math.tanh(sigma_x) + 1) / 2, 0.01 + (tf.math.tanh(sigma_y) + 1) / 2
        # sigma_x, sigma_y = 14 / 200 * tf.ones_like(mu_x), 14 / 120 * tf.ones_like(mu_y)
        # sigma_x, sigma_y = 0.50 * tf.ones_like(sigma_x), 0.50 * tf.ones_like(sigma_y)
        mu, sigma = tf.stack([mu_x, mu_y], axis=-1), tf.stack([sigma_x, sigma_y], axis=-1)
        return [[mu[:, i, :], sigma[:, i, :]] for i in range(self.blocks)]


class AlexGeneralisedClassifier(tf.keras.Model):

    def __init__(self, backbone, size=128, N=10, conv=True):
        super().__init__()
        self.backbone = backbone
        self.N = N
        self.size = size
        self.indextensor_x = tf.constant([i / 200 for i in range(200)])
        self.indextensor_y = tf.constant([i / 120 for i in range(120)])

        self.modules = [self.backbone]
        if conv:
            self.modules.append(Conv2D(self.N, 5))
            self.modules.append(GlobalMaxPooling2D())
            self.modules.append(Softmax())
        else:
            self.modules.append(Flatten())
            self.modules.append(Dense(size, activation=ReLU()))
            self.modules.append(Dropout(0.5))
            self.modules.append(Dense(size, activation=ReLU()))
            self.modules.append(Dropout(0.5))
            self.modules.append(Dense(self.N, activation=Softmax()))

        self.model = tf.keras.Sequential(self.modules)

    def call(self, x, params, **kwargs):
        """
        This classifier uses a continuous distribution to efficiently compute attention and do so in an interpretable
        way. Interpretable since the attention is directly visible as a distribution.
        We can integrate this approach with the RoI concept, predicting RoIs in the feature map. However, this appraoch
        would not be as interpretable.
        """
        mu, sigma = params

        mu_x, sigma_x = mu[:, 0], sigma[:, 0]
        mu_x, sigma_x = tf.expand_dims(mu_x, axis=-1), tf.expand_dims(sigma_x, axis=-1)
        mu_y, sigma_y = mu[:, 1], sigma[:, 1]
        mu_y, sigma_y = tf.expand_dims(mu_y, axis=-1), tf.expand_dims(sigma_y, axis=-1)

        distrx = tfp.distributions.GeneralizedNormal(loc=mu_x, scale=sigma_x, power=8.)
        distry = tfp.distributions.GeneralizedNormal(loc=mu_y, scale=sigma_y, power=8.)
        distrx_max = distrx.prob(mu_x)
        distry_max = distry.prob(mu_y)

        mask_cols = distrx.prob(self.indextensor_x) / distrx_max
        mask_cols = tf.matmul(tf.expand_dims(mask_cols, axis=-1), tf.ones([1, 120]))
        maskx = tf.expand_dims(tf.transpose(mask_cols, perm=[0, 2, 1]), axis=-1)

        mask_rows = distry.prob(self.indextensor_y) / distry_max
        mask_rows = tf.matmul(tf.expand_dims(mask_rows, axis=-1), tf.ones([1, 200]))
        masky = tf.expand_dims(mask_rows, axis=-1)

        masked_image = x * maskx * masky

        x = self.model.call(masked_image)
        return x

class AlexBaseline(tf.keras.Model):

    def __init__(self, digits, size=512):
        super().__init__()
        self.backbone = AlexConvolutions()
        self.regressor = AlexRegressor(self.backbone, blocks=digits)
        self.classifier = AlexGeneralisedClassifier(self.backbone)

    def call(self, x, **kwargs):
        regions = self.regressor(x)
        classes = []
        for region in regions:
            classes.append(self.classifier(x, region))
        return (classes, [i[0] for i in regions])