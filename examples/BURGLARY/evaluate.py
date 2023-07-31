import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from examples.BURGLARY.data import datasets as dts


def evaluate_plogic(test_set, model, plot=False):
    test_queries = test_set.to_queries()
    MSE = []
    for q in test_queries:
        answer = torch.tensor(list(model.solve([q])[0].result.values())[0]).item()
        MSE.append(answer - q.p)
    MSEres = np.mean(MSE)

    if plot:
        plt.hist(MSE, bins=100, range=(-0.3, 0.3))
        plt.show()
    return MSEres

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

def means_and_variances(param_net, batch_size):
    I = tf.ones([batch_size, 1])
    result = param_net.call(I)
    result = [list(i.numpy().round(3)) for i in result]
    return result
