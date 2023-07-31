import math
import tensorflow as tf

from typing import Optional
from semiring import Semiring, Result
from problog.logic import Constant, Term, term2list, list2term


class TensorSemiring(Semiring):
    def __init__(self, model, substitution, values):
        Semiring.__init__(self, model, substitution, values)

    def negate(self, a):
        return 1.0 - a

    def one(self):
        return 1.

    def zero(self):
        return 0.

    def plus(self, a, b):
        return a + b

    def times(self, a, b):
        if type(a) is float or type(b) is float:
            return a * b
        if len(a.shape) == 3 and len(b.shape) == 2:
            mult = tf.expand_dims(tf.ones_like(tf.reduce_sum(a, axis=[-1, -2])), axis=-1)
            b = tf.expand_dims(tf.matmul(mult, b), axis=1)
        if len(b.shape) == 3 and len(a.shape) == 2:
            mult = tf.expand_dims(tf.ones_like(tf.reduce_sum(b, axis=[-1, -2])), axis=-1)
            a = tf.expand_dims(tf.matmul(mult, a), axis=1)
        result = a * b
        return result

    def value(self, a, key=None):
        if type(a) is Constant:
            return tf.constant(float(a))
        elif type(a) is float:
            return tf.constant(a)
        elif type(a) is Term:
            if a.functor == "nn":
                net, inputs = a.args[0], a.args[1]
                inputs = inputs.apply_term(self.substitution)
                val = self.values[(net, inputs)]
                if len(a.args) == 3:
                    domain = term2list(a.args[2], False)
                    domain = tf.expand_dims(tf.stack(domain), axis=-1)
                    batchv = tf.expand_dims(tf.ones_like(tf.reduce_sum(val, axis=-1)), axis=0)
                    D = tf.transpose(tf.matmul(domain, batchv))
                    return tf.stack([D, val])
                else:
                    return val
            elif a.functor == "t":
                i = int(a.args[0])
                p = tf.constant(self.model.parameters[i], requires_grad=True)
                return p
            elif a.functor == "tensor":
                return self.model.get_tensor(a)
            elif a.functor == "'/'":  # Deals with probabilities formatted as franctions
                return tf.constant(float(a))
            else:
                raise Exception("unhandled term {}".format(a.functor))
        else:
            return tf.constant(float(a.compute_value()))

    def is_one(self, a):
        return False

    def is_zero(self, a):
        return False

    def is_dsp(self):
        return True

    def normalize(self, a, z):
        if self.is_one(z):
            return a
        # print('normalizing with ', self.one()-float(z))
        return a / z

    @staticmethod
    def cross_entropy(
        result: Result,
        target: float,
        weight: float,
        q: Optional[Term] = None,
        eps: float = 1e-12,
    ) -> float:

        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        if type(p) is float:
            print("This should not be happening.")
            loss = (
                -(target * math.log(p + eps) + (1.0 - target) * math.log(1.0 - p + eps))
                * weight
            )
        else:
            if target == 1.0:
                loss = -tf.math.log(p) * weight
            elif target == 0.0:
                loss = -tf.math.log(1.0 - p) * weight
            else:
                loss = (
                    -(
                        target * tf.math.log(p + eps)
                        + (1.0 - target) * tf.math.log(1.0 - p + eps)
                    )
                    * weight
                )
            loss = tf.reduce_mean(loss)
            loss.backward(retain_graph=True)
            # loss.backward()
        return float(loss)

    @staticmethod
    def cross_entropy_regulariser(result: Result,
        target: float,
        weight: float,
        q: Optional[Term] = None,
        eps: float = 1e-12
    ) -> float:

        nn_values = result.semiring.values
        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        if type(p) is float:
            loss = (
                    -(target * math.log(p + eps) + (1.0 - target) * math.log(1.0 - p + eps))
                    * weight
            )
        else:
            if target == 1.0:
                loss = -tf.math.log(p) * weight
            elif target == 0.0:
                loss = -tf.math.log(1.0 - p) * weight
            else:
                loss = (
                        -(
                                target * tf.math.log(p + eps)
                                + (1.0 - target) * tf.math.log(1.0 - p + eps)
                        )
                        * weight
                )

            def log(tensor, base):
                return tf.experimental.numpy.log10(tensor) / tf.experimental.numpy.log10(base)

            def Hn(values, N):
                print('predicted values', values)
                entropy_op = values * log(values, N)
                return - tf.reduce_sum(entropy_op)

            # print('This is the loss', loss)
            for key, value in nn_values.items():
                if 'class' in key[0].functor:
                    entropy = Hn(value, tf.constant(value.shape[0]))
                    reg = 8 * entropy
                    # print('Regulariser', reg)
                    loss += reg
            # print('Loss after reg', loss)
            loss.backward(retain_graph=True)
        return float(loss)

    @staticmethod
    def mse(
        result: Result, target: float, weight: float, q: Optional[Term] = None
    ) -> float:

        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        loss = (p - target) ** 2 * weight
        if type(p) is not float:
            loss.backward(retain_graph=True)
        return float(loss)
