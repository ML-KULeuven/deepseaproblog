from __future__ import annotations

import tensorflow as tf

from typing import Optional
from problog.logic import Term


def get_tensor_function(network: Network):
    def tensor_function(*args):
        return tuple(network.model.get_tensor(arg) for arg in args)
    return tensor_function


class Network(object):
    def __init__(
        self,
        network_module: tf.keras.Model,
        name: str,
        k: Optional[int] = None,
    ):
        """

        :param network_module: The neural network module.
        :param name: The name of the network as used in the neural predicate nn(name, ...)
        :param k: If k is set, only the top k results of the neural network will be used.
        Otherwise, they are evaluated one by one.
        """
        self.network_module = network_module
        self.name = name
        self.function = get_tensor_function(self)
        self.model = None
        self.n = 0
        self.domain = None
        self.cache = dict()
        self.k = k
        self.eval_mode = False
        self.det = False

    def __call__(self, to_evaluate: list) -> list:
        """
        Evaluate the network on the given inputs.
        :param to_evaluate: A list that contains the inputs that the neural network should be evaluated on.
        :return:
        """
        next_to_evaluate = []
        for e in to_evaluate:
            decomposition = self.proper_decomposition(e)
            next_to_evaluate.append(decomposition)
        evaluated = [self.network_module(*e) for e in next_to_evaluate]
        return evaluated

    def proper_decomposition(self, to_evaluate):
        decomposition = []
        for e in to_evaluate:
            if type(e) is Term and e.functor == 'tensor':
                decomposition.append(*self.function(e))
            elif type(e) is list:
                decomposition.append(self.proper_decomposition(e))
            elif type(e) is float:
                decomposition.append(tf.constant(e, dtype=tf.float32))
            elif type(e) is int:
                decomposition.append(tf.constant(e, dtype=tf.int64))
            else:
                decomposition.append(e)
        return decomposition

    def get_hyperparameters(self):
        parameters = {
            "name": self.name,
            "module": str(self.network_module),
            "k": self.k,
        }
        return parameters


class PCF(Network):

    def __init__(self,
                 network_module: tf.keras.Model,
                 inputs: int, name: str,
                 k: Optional[int] = None):
        super().__init__(network_module, name, k)
        self.inputs = inputs
