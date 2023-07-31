import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from examples.BURGLARY.data import datasets as dts

def evaluate_class_network(net, restrictions, dataset, reverse=False, batch_size=10):
    acc = 0
    count = 0
    its = 0
    I_batch_list = []
    T_batch_list = []
    max = 1000
    for I, T in dts['test']:
        if T in restrictions:
            if dataset == 'val' and its > max:
                break
            elif dataset == 'test' and its < max:
                pass
            I_batch_list.append(I)
            T_batch_list.append(torch.tensor(T))

            if its % batch_size == 9:
                I_batch = torch.stack(I_batch_list)
                T_batch = torch.stack(T_batch_list)

                I = tf.reshape(tf.constant(I_batch.numpy()), [batch_size, 28, 28, 1])
                T = tf.constant(T_batch.numpy())
                O = tf.argmax(net.call(I), axis=1)
                if reverse:
                    acc += tf.reduce_sum(tf.where(((O + 8) == T), 1, 0))
                else:
                    acc += tf.reduce_sum(tf.where((T == O), 1, 0))
                count += batch_size
                I_batch_list = []
                T_batch_list = []
            its += 1
    result = acc / count
    return float(result)


def evaluate_temperature(regressionnet, set, loss="MSE"):
    val, counter = 0, 0
    for I, T in set:
        I1, I2, X, T, N1, N2 = I
        T = tf.cast(T, tf.float32)
        pred = regressionnet.call(X)
        if loss == "MSE":
            val += tf.reduce_mean(tf.math.squared_difference(pred[0], T))
        else:
            val += tf.reduce_mean(tf.math.abs(T - pred[0]))
        counter += X.shape[0]
    return float(val / counter)

def evaluate_noise(regressionnet):
    return float(tf.abs(tf.math.exp(regressionnet.noise) - 3.))
