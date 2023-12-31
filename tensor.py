from typing import Optional, Union
from tensorflow import Tensor
from problog.logic import Term

import tensorflow as tf


TensorStoreIndex = Union[int, Term]

class TensorStore(object):
    """
    A class that allows tensor to be stored and associated with a unique identifier.
    """

    def __init__(self):
        self.tensor_store = dict()
        self.i = 0

    def store(
        self, tensor: Tensor, i: Optional[TensorStoreIndex] = None
    ) -> TensorStoreIndex:
        """
        Stores a tensor.
        :param tensor: The tensor to store.
        :param i: If given, then tensor will be stored with index i, otherwise, a unique integer will be used.
        :return: The unique identifier of the stored tensor.
        """
        if i is None:
            i = self.i
            self.i += 1
        self.tensor_store[i] = tensor
        return i

    def __getitem__(self, item):
        return self.tensor_store[item]

    def __contains__(self, item):
        return item in self.tensor_store

    def clear(self):
        """
        Clear the tensor store.
        :return:
        """
        self.tensor_store.clear()
        self.i = 0


ModuleStoreKey = [
    str,
]


class ModuleStore(object):
    def __init__(self):
        self.module_store = dict()

    @property
    def keys(self):
        return self.module_store.keys()

    def store(self, module: tf.keras.Model, key: ModuleStoreKey):
        self.module_store[key] = module

    def __getitem__(self, key):
        return self.module_store[key]

    def __contains__(self, key):
        return key in self.module_store

    def clear(self):
        self.module_store.clear()
