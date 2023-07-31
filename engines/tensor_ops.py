import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class DistanceOp(tf.keras.layers.Layer):

    def __init__(self):
        super(DistanceOp, self).__init__()

    def call(self, x, y):
        """
        At least one of x or y needs to be a multivariate variable, otherwise distance
        does not make sense. Dimension of variable should be at axis -2.
        """
        return tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(y - x), axis=-2)), axis=-2)

class Sampler(tf.keras.layers.Layer):

    def __init__(self, model, sample_nb=250, temperature=2., alpha=1e-4, scheme="tanh"):
        super(Sampler, self).__init__()
        self.model = model
        self.sample_nb = sample_nb
        self.temperature = temperature
        self.counter = tf.Variable(0, trainable=False, name="SamplerCounter")
        self.alpha = alpha
        self.scheme = scheme

    def annealed_temperature(self):
        if self.scheme == "tanh":
            x = 5. + (self.temperature - 5.) * tf.math.tanh(tf.cast(self.counter, dtype=tf.float32) * self.alpha)
        else:
            x = self.temperature
        return x

    def gumbel_straight_through(self, d, classes):
        soft_out = d.sample(sample_shape=self.sample_nb)
        hard_out = tf.one_hot(tf.argmax(soft_out, axis=-1), soft_out.shape[-1])
        """ The straight-through trick """
        diff_out = tf.stop_gradient(hard_out - soft_out) + soft_out
        classes = tf.expand_dims(classes, axis=-1)
        return tf.transpose(tf.matmul(diff_out, classes), perm=[1, 2, 0])

    def categorical_domain(self, samples, classes):
        return tf.gather(classes, samples)

    def create_distribution(self, params, distribution):
        params = [tf.cast(param, dtype=tf.float32) for param in params]
        if type(distribution) == list:
            params = [param * distribution[1] for param in params]
            distribution = distribution[0]
        if distribution != "gumbelsoftmax" and distribution != "categorical":
            for id, param in enumerate(params):
                params[id] = tf.convert_to_tensor(param, dtype=tf.float32)
        if distribution == "normal":
            mu, sigma = params
            d = tfp.distributions.Normal(loc=mu, scale=sigma)
        elif distribution == "generalisednormal":
            mu, sigma, power = params
            d = tfp.distributions.GeneralizedNormal(loc=mu, scale=sigma, power=power)
        elif distribution == "multivariatenormal":
            mu, cov = params
            d = tfp.distributions.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
        elif distribution == "gamma":
            alpha, beta = params
            d = tfp.distributions.Gamma(concentration=alpha, rate=beta)
        elif distribution == "beta":
            alpha, beta = params
            d = tfp.distributions.Beta(concentration1=alpha, concentration0=beta)
        elif distribution == "dirichlet":
            d = tfp.distributions.Dirichlet(concentration=params)
        elif distribution == "exponential":
            d = tfp.distributions.Exponential(rate=params)
        elif distribution == "laplace":
            mu, sigma = params
            d = tfp.distributions.Laplace(loc=mu, scale=sigma)
        elif distribution == "logistic":
            mu, sigma = params
            d = tfp.distributions.Logistic(loc=mu, scale=sigma)
        elif distribution == "poisson":
            d = tfp.distributions.Poisson(rate=params)
        elif distribution == "chi":
            d = tfp.distributions.Chi(params)
        elif distribution == "student":
            df, mu, sigma = params
            d = tfp.distributions.StudentT(df, mu, sigma)
        elif distribution == "uniform":
            low, high = params
            d = tfp.distributions.Uniform(low, high)
        elif distribution == "categorical":
            """ Note: samples of this distribution are not differentiable! """
            probs, _ = params
            d = tfp.distributions.Categorical(probs=probs)
        elif distribution == 'bernoulli':
            probs, _ = params
            d = tfp.distributions.Bernoulli(probs=probs)
        elif distribution == "gumbelsoftmax":
            logits, _ = params
            d = tfp.distributions.RelaxedOneHotCategorical(temperature=self.annealed_temperature(), logits=logits)
        else:
            d = None
            Exception(f"The distribution {distribution} is not yet supported!")
        return d

    def call(self, params, distribution):
        d = self.create_distribution(params, distribution)
        if distribution == "gumbelsoftmax":
            out = self.gumbel_straight_through(d, params[1])
        else:
            samples = d.sample(sample_shape=self.sample_nb)
            if len(samples.shape) == 1:
                out = tf.expand_dims(tf.expand_dims(samples, axis=0), axis=0)
            elif len(samples.shape) == 2:
                out = tf.expand_dims(tf.transpose(samples, perm=[1, 0]), axis=0)
            elif len(samples.shape) == 3:
                out = tf.transpose(samples, perm=[1, 2, 0])
            else:
                out = samples
        """ The shape of all samples is (b, p, dim, samples). PCFs reduce it to (b, p, samples). """
        if len(out.shape) == 3:
            out = tf.expand_dims(out, axis=1)
        return out


class Operator(tf.keras.Model):

    def __init__(self, operator, name=None):
        super(Operator, self).__init__()
        self.operator = operator

    def op(self, x, y):
        if self.operator == 'sub':
            return x - y
        elif self.operator == 'add':
            return x + y
        elif self.operator == 'mul':
            return x * y
        elif self.operator == 'smaller':
            return x < y
        elif self.operator == "div":
            return x / y
        elif self.operator == "rounddiv":
            return x // y
        elif isinstance(self.operator, tf.keras.Model):
            return self.operator(x, y)
        else:
            ValueError("Operator not implemented yet!")

    def call(self, x, y):
        scalar_x = False
        scalar_y = False
        try:
            if len(x.shape) == 0:
                x = [[x], tf.constant([1.])]
                scalar_x = True
        except Exception:
            pass
        try:
            if len(y.shape) == 0:
                y = [[y], tf.constant([1.])]
                scalar_y = True
        except Exception:
            pass
        
        if type(x) is float:
                x = [[tf.cast(x, dtype=tf.int64)], tf.constant([1.])]
                scalar_x = True
        if type(y) is float:
                y = [[tf.cast(y, dtype=tf.int64)], tf.constant([1.])]
                scalar_y = True

        if not scalar_x:
            batch_mult = tf.transpose(tf.ones_like(tf.reduce_sum(x[1], axis=-1)))
        elif not scalar_y:
            batch_mult = tf.transpose(tf.ones_like(tf.reduce_sum(y[1], axis=-1)))
        else:
            return self.op(x[0][0], y[0][0])

        x_p, y_p = tf.expand_dims(x[1], axis=-1), tf.expand_dims(y[1], axis=-2)
        matrix = tf.matmul(x_p, y_p)

        combinations = np.zeros([len(x[0]), len(y[0])], dtype=np.int64)
        for iid, i in enumerate(x[0]):
            for jid, j in enumerate(y[0]):
                combinations[iid, jid] = self.op(int(i), int(j))
        options = [tf.cast(i, dtype=tf.int64) for i in  np.unique(combinations)]
        combinations = tf.constant(combinations)

        C = tf.expand_dims(combinations, -1)
        M = tf.transpose(tf.matmul(C, tf.expand_dims(tf.cast(batch_mult, dtype=tf.int64), axis=0)), perm=[2, 0, 1])

        outcome_probs = []

        for n in options:
            outcome_probs.append(tf.reduce_sum(tf.where((M == n), matrix, 0), axis=(-2, -1)))
        probs = tf.transpose(tf.stack(outcome_probs))
        return [options, probs]


class ContinuousOperation(Operator):

    def __init__(self, operator, name=None):
        super(ContinuousOperation, self).__init__(operator, name)

    def op(self, x, y):
        result = super().op(x, y)
        if self.operator == "rounddiv":
            result = tf.stop_gradient(result - x / y) + x / y
        return result

    def call(self, x, y):
        x = standardise_tensor(x)
        y = standardise_tensor(y)
        result = self.op(x, y)
        return result


# class DiscreteOperation(Operator):
#
#     def __init__(self, operator, name=None, model=None):
#         super(DiscreteOperation, self).__init__(operator, name)
#         self.model = model
#
#     def call(self, x, y):
#         # x = tf.cast(x, dtype=tf.int64)
#         # y = tf.cast(y, dtype=tf.int64)
#
#         # x = tf.cast(x, dtype=tf.float32)
#         # y = tf.cast(y, dtype=tf.float32)
#         if len(x.shape) == 0:
#             x = tf.expand_dims(x, axis=0)
#         if len(y.shape) == 0:
#             y = tf.expand_dims(y, axis=0)
#         while(len(x.shape) < 4):
#             x = tf.expand_dims(x, axis=1)
#         while(len(y.shape) < 4):
#             y = tf.expand_dims(y, axis=1)
#
#         # if x.shape[1] > 1 and y.shape[1] > 1:
#         #     x_samples, y_samples = tf.expand_dims(x[:, 0, :, :], axis=1),  tf.expand_dims(y[:, 0, :, :], axis=1)
#         #     x_conditional, y_conditional = x[:, 1:, :, :], y[:, 1:, :, :]
#         #     samples_result = self.op(x_samples, y_samples)
#         #     x_conditional_result = self.op(x_conditional, y_samples)
#         #     y_conditional_result = self.op(x_samples, y_conditional)
#         #     result = tf.concat([samples_result, x_conditional_result, y_conditional_result], axis=1)
#         #     self.model.conditionals[result.ref()] = self.model.conditionals[x.ref()] + self.model.conditionals[y.ref()]
#         # elif x.shape[1] > 1:
#         #     result = self.op(x, y)
#         #     self.model.conditionals[result.ref()] = self.model.conditionals[x.ref()]
#         # elif y.shape[1] > 1:
#         #     result = self.op(x, y)
#         #     self.model.conditionals[result.ref()] = self.model.conditionals[y.ref()]
#         # else:
#         #     result = self.op(x, y)
#         # return result
#
#         return self.op(x, y)


class Is(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super(Is, self).__init__()

    def call(self, target, options):
        pos_outcomes = tf.cast(tf.stack(options[0], axis=0), dtype=tf.float32)
        outcome_probs = options[1]

        batch_mult = tf.expand_dims(tf.ones_like(tf.reduce_sum(outcome_probs, axis=-1)), axis=-1)
        outcomes_reshape = tf.matmul(batch_mult, tf.expand_dims(pos_outcomes, axis=0))
        subtraction = tf.subtract(outcomes_reshape, tf.cast(target, dtype=tf.float32)) == 0.

        result = tf.reduce_sum(tf.where(subtraction, outcome_probs * 1, outcome_probs * 0), axis=-1)
        return tf.expand_dims(tf.expand_dims(result, axis=-1), axis=-1)


class SmallerThan(tf.keras.layers.Layer):

    def __init__(self, alpha=1e-3, minmult=1., maxmult=1., scheme="constant"):
        super(SmallerThan, self).__init__()
        self.counter = tf.Variable(0, trainable=False, name="SmallerThanCounter")
        self.alpha = alpha
        self.minmult = minmult
        self.maxmult = maxmult
        self.scheme = scheme

    def multiplier(self):
        if self.scheme == "tanh":
            x = self.minmult + (self.maxmult - self.minmult) * tf.math.tanh(tf.cast(self.counter, dtype=tf.float32) * self.alpha)
        elif self.scheme == "sigmoid":
            x = self.minmult + (self.maxmult - self.minmult) * tf.math.sigmoid(tf.cast(self.counter, dtype=tf.float32) * self.alpha - 0.5)
        else:
            x = self.maxmult
        return x

    def call(self, x, y):
        self.counter.assign_add(1)
        x = standardise_tensor(x)
        y = standardise_tensor(y)
        hard_out = tf.where(y > x, 1., 0.)
        soft_out = tf.math.sigmoid(self.multiplier() * (y - x))
        out = tf.stop_gradient(hard_out - soft_out) + soft_out
        if len(out.shape) == 4:
            out = tf.reduce_prod(out, axis=-2)
        return out


def standardise_tensor(x):
    if x.dtype != tf.float32:
        x = tf.cast(x, dtype=tf.float32)
    while len(x.shape) < 3:
        x = tf.expand_dims(x, axis=-1)
    return x

class ExpLnLoss(tf.keras.losses.Loss):
    """ Implementation of L_p^p norm as loss function. """
    def __init__(self, n=1):
        super(ExpLnLoss, self).__init__()
        self.n = n

    def call(self, diff):
        if self.n == np.inf:
            return tf.norm(diff, ord=self.n, axis=-1)
        return tf.reduce_mean(tf.pow(tf.abs(diff), self.n), axis=-1)


class SquaredSech(tf.keras.losses.Loss):
    """ Implementation the sech^2 loss function with sigmoids """
    def call(self, diff):
        return 4 * tf.sigmoid(2 * diff) * tf.sigmoid(-2 * diff)


class Equals(tf.keras.layers.Layer):

    def __init__(self, alpha=1e-3, minmult=1., maxmult=1., scheme="constant"):
        super(Equals, self).__init__()
        self.counter = tf.Variable(0, trainable=False, name="EqualsCounter")
        self.alpha = alpha
        self.minmult = minmult
        self.maxmult = maxmult
        self.scheme = scheme
        self.loss = SquaredSech()

    def multiplier(self):
        if self.scheme == "tanh":
            x = self.minmult + (self.maxmult - self.minmult) * tf.math.tanh(tf.cast(self.counter, dtype=tf.float32) * self.alpha)
        else:
            x = self.maxmult
        return x

    def call(self, x, y):
        self.counter.assign_add(1)
        x = standardise_tensor(x)
        y = standardise_tensor(y)
        x = tf.sort(x, axis=-1)
        y = tf.sort(y, axis=-1)
        diff = self.multiplier() * (y - x)
        if len(diff.shape) == 4:
            diff = tf.reduce_sum(diff, axis=-2)
        loss = self.loss.call(diff)
        return loss

class SoftUnification(tf.keras.layers.Layer):

    def __init__(self, alpha=1e-3, minmult=1., maxmult=1., scheme="tanh", type="Ln", n=1):
        super(SoftUnification, self).__init__()
        self.alpha = alpha
        self.maxmult = maxmult
        self.minmult = minmult
        self.scheme = scheme
        self.type = type
        self.n = n
        if self.type == "Ln":
            self.loss = ExpLnLoss(self.n)
        else:
            self.loss = SquaredSech()
        self.counter = tf.Variable(0, trainable=False, name="ReconLossCounter")

    def multiplier(self):
        if self.scheme == "tanh" and tf.cast(self.counter, dtype=tf.float32) < 1 / self.alpha:
            x = self.minmult + (self.maxmult - self.minmult) * tf.math.tanh(
                tf.cast(self.counter, dtype=tf.float32) * self.alpha)
        else:
            x = self.maxmult
        return x

    def call(self, x, y):
        self.counter.assign_add(1)
        x = tf.reshape(x, shape=[-1, math.prod(x.shape[1:])])
        y = tf.reshape(y, shape=[-1, math.prod(y.shape[1:])])
        loss = self.multiplier() * self.loss.call(y - x)
        while len(loss.shape) < 3:
            loss = tf.expand_dims(loss, axis=-1)
        return tf.exp(-loss)

