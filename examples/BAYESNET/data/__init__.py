import random
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import scipy.io as io
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from random import randint, random
from random import seed as sd
from examples.BURGLARY.data import create_dataloaders as b_loaders


_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.CIFAR10(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.CIFAR10(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def expected_output(t_data, c_data, prob=False, noise=3, full=0, batch_size=10, dataset_size=np.inf):
    t_len = len(t_data)
    joined_data = []
    t_max = 19.3  # used to be 38.9
    t_min = -9.7  # used to be 11.3
    if prob:
        true_pleasant00 = tf.expand_dims(tfd.Beta(9, 2).sample(1000), axis=-1)
        true_pleasant01 = tf.expand_dims(tfd.Beta(1, 1).sample(1000), axis=-1)
        true_pleasant10 = tf.expand_dims(tfd.Beta(11, 7).sample(1000), axis=-1)
        true_pleasant11 = tf.expand_dims(tfd.Beta(1, 9).sample(1000), axis=-1)
    else:
        true_pleasant00 = tfd.Beta(9, 2).sample(1)[0]
        true_pleasant01 = tfd.Beta(1, 1).sample(1)[0]
        true_pleasant10 = tfd.Beta(11, 7).sample(1)[0]
        true_pleasant11 = tfd.Beta(1, 9).sample(1)[0]
    for id, i in enumerate(c_data):
        if id > dataset_size // batch_size:
            break
        ids = np.random.randint(0, t_len - 1, size=batch_size)
        x_batch = []
        t_batch = []
        for id in ids:
            x_batch.append(tf.constant(t_data[id][0]))
            t_inter = (t_data[id][1][0] + t_data[id][2][0]) / 2
            t_batch.append(t_inter * (t_max - t_min) + t_min)
        true_temp = tfd.Normal(t_batch, noise)
        cloudy, humid = i[2], i[3]
        rainy = tf.where(tf.logical_and(cloudy != 0, humid - 8 == 1), 1., 0.)
        depressed = tf.where(cloudy == 2, 0.4, 0.)
        depressed += tf.where(cloudy == 1, 0.2, 0.)
        if prob:
            temp_samples = true_temp.sample(1000)
        else:
            temp_samples = true_temp.sample(1000)
        rcond = tf.squeeze(tf.where(temp_samples < 0, 1., 0.))
        # nrcond = tf.squeeze(tf.where(tf.logical_and(temp_samples > 15, temp_samples < 25), 1., 0.))
        nrcond = tf.squeeze(tf.where(15 < temp_samples, 1., 0.))

        pleasant_samples = tf.expand_dims(rainy, axis=-1) * tf.transpose(rcond * true_pleasant10 + (1 - rcond) * true_pleasant11) + \
                           tf.expand_dims(1 - rainy, axis=-1) * tf.transpose(nrcond * true_pleasant00 + (1 - rcond) * true_pleasant01)

        if prob:
            good_day = tf.reduce_mean(tf.expand_dims(1 - depressed, axis=-1) * tf.where(pleasant_samples > 0.5, 1., 0.), axis=-1)
        else:
            good_day_prob = tf.reduce_mean(tf.expand_dims(1 - depressed, axis=-1) * tf.where(pleasant_samples > 0.5, 1., 0.), axis=-1)
            good_day = tfd.Bernoulli(good_day_prob).sample([1])[0]

        # temp_cond = tf.squeeze(tf.where(tf.logical_or(temp_samples > 15, temp_samples < 0), 1., 0.))
        # if prob:
        #     prob_samples = true_temp.sample(1000)[:, :, 0]
        #     temp_cond = tf.reduce_mean(tf.where(prob_samples > 15, 1., 0.), axis=0)
        if full == 2:
            joined_data.append([[i[0], i[1], tf.squeeze(tf.stack(x_batch)), tf.squeeze(tf.stack(t_batch)), cloudy, humid], tf.ones_like(good_day)])
        elif full == 1:
            joined_data.append(
                [[i[0], i[1], tf.squeeze(tf.stack(x_batch))],
                 good_day])
        else:
            joined_data.append([[i[0], i[1], tf.squeeze(tf.stack(x_batch))], good_day])
    return joined_data

def expected_output_simplified(t_data, c_data, prob, noise=3, full=3, batch_size=10, dataset_size=np.inf):
    t_len = len(t_data)
    joined_data = []
    t_max = 19.3  # used to be 38.9
    t_min = -9.7  # used to be 11.3
    if prob:
        snowy_samples = tf.expand_dims(tfd.Beta(2, 7).sample(1000), axis=-1)
        sunny_samples = tf.expand_dims(tfd.Beta(5, 3).sample(1000), axis=-1)
    else:
        snowy_samples = tfd.Beta(2, 7).sample(1)[0]
        sunny_samples = tfd.Beta(5, 3).sample(1)[0]
    for id, i in enumerate(c_data):
        if id > dataset_size // batch_size:
            break
        ids = np.random.randint(0, t_len - 1, size=batch_size)
        x_batch = []
        t_batch = []
        for id in ids:
            x_batch.append(tf.constant(t_data[id][0]))
            t_inter = (t_data[id][1][0] + t_data[id][2][0]) / 2
            t_batch.append(t_inter * (t_max - t_min) + t_min)
        true_temp = tfd.Normal(t_batch, noise)
        cloudy, humid = i[2], i[3]
        rainy = tf.where(tf.logical_and(cloudy != 0, humid - 8 == 1), 1., 0.)
        depressed = tf.where(cloudy == 2, 0.4, 0.)
        depressed += tf.where(cloudy == 1, 0.2, 0.)
        if prob:
            temp_samples = true_temp.sample(1000)
        else:
            temp_samples = true_temp.sample(1000)
        rcond = tf.squeeze(tf.where(temp_samples < 0, 1., 0.))
        nrcond = tf.squeeze(tf.where(15 < temp_samples, 1., 0.))
        snowy = rainy * rcond * snowy_samples
        sunny = (1 - rainy) * nrcond * sunny_samples

        snowy_sat = tf.where(snowy < 0.3, 1., 0.)
        sunny_sat = tf.where(sunny > 0.6, 1., 0.)
        good_day_prob = tf.reduce_mean((1 - depressed) * (snowy_sat + sunny_sat), axis=0)
        if prob:
            good_day = good_day_prob
        else:
            good_day = tfd.Bernoulli(good_day_prob).sample([1])[0]

        if full == 2:
            joined_data.append([[i[0], i[1], tf.squeeze(tf.stack(x_batch)), tf.squeeze(tf.stack(t_batch)), cloudy, humid], tf.ones_like(good_day)])
        elif full == 1:
            joined_data.append(
                [[i[0], i[1], tf.squeeze(tf.stack(x_batch))], good_day])
        else:
            joined_data.append([[i[0], i[1], tf.squeeze(tf.stack(x_batch))], good_day])

def create_dataloaders(dataset_name, restrictions, batch_size=10, seed=7,
                       noise=3, prob=False, full=0, simplified=False, dataset_size=np.inf):
    data = b_loaders(dataset_name, restrictions, batch_size=batch_size, direct=True)

    number_dataset_realizations = 1
    number_weight_initializations = 4
    train_size = [0.75, 0.5, 0.25, 0.1]
    t_train_data, t_val_data, t_test_data = t_loaders(seed, 1, number_weight_initializations,
                                                number_dataset_realizations, train_size, 0.1, 0.1)
    t_data = {}
    t_data["train"] = list(t_train_data.as_numpy_iterator())
    t_data["val"] = list(t_val_data.as_numpy_iterator())
    t_data["test"] = list(t_test_data.as_numpy_iterator())

    if simplified:
        return expected_output_simplified(t_data[dataset_name], data, prob, noise, full, batch_size, dataset_size)
    return expected_output(t_data[dataset_name], data, prob, noise, full, batch_size, dataset_size)


def prep_data(prep_directory, prep_name, prep_ext):
    """Load in the data.
    Parameters
    ----------
    prep_directory : string
        String denoting the directory in which the dataset is saved.
    prep_name : string
        String denoting the name of the file in which the dataset is saved.
    prep_ext : string
        String denoting the extension of the savefile. Supported extension are .mat and .csv.
    Returns
    ----------
    prep_input : np.ndarray
        Numpy array containing the input of all the instances.
    prep_output : np.ndarray
        Numpy array containing the output of all the instances.
    """
    if prep_ext == '.mat':
        prep_data_loaded = io.loadmat(prep_directory + prep_name)
        prep_input = prep_data_loaded.get('Imx')
        prep_output = prep_data_loaded.get('Omx')
    elif prep_ext == '.csv':
        prep_data_loaded = pd.read_csv(prep_directory + prep_name + '.csv', index_col=[0, 1], header=0)
        prep_data_loaded = prep_data_loaded.transpose()
        prep_input = prep_data_loaded['Input'].to_numpy()
        prep_output = prep_data_loaded['Output'].to_numpy()
    else:
        print('Extension is not supported.')
        prep_input = []
        prep_output = []

    return prep_input, prep_output

def split_train_test_val(split_input_data, split_output_data, split_test_size, split_val_size, split_random_state1, split_random_state2):
    """Split the dataset into train, validation and test sets using the pseudo random seeds split_random_state1 and split_random_state2.
    Parameters
    ----------
    split_input_data : np.ndarray
        Numpy array containing the input of all instances in the dataset.
    split_output_data : np.ndarray
        Numpy array containing the output of all instances in the dataset.
    split_test_size : float
        Float in (0,1) denoting the test size relative to the size of the total dataset.
    split_val_size : float
        Float in (0,1) denoting the validation size relative to the size of the total dataset.
    split_random_state1 : int
        Integer denoting the pseudo random seed used to determine the test set.
    split_random_state2 : int
        Integer denoting the pseudo random seed used to determine the train and validation set.
    Returns
    ----------
    split_train_input : np.ndarray
        Numpy array containing the input of all instances in the train set.
    split_val_input : np.ndarray
        Numpy array containing the input of all instances in the validation set.
    split_test_input : np.ndarray
        Numpy array containing the input of all instances in the test set.
    split_train_output : np.ndarray
        Numpy array containing the output of all instances in the train set.
    split_val_output : np.ndarray
        Numpy array containing the output of all instances in the validation set.
    split_test_output : np.ndarray
        Numpy array containing the output of all instances in the test set.
    """

    split_train_val_input, split_test_input, split_train_val_output, split_test_output = \
        train_test_split(split_input_data,
                         split_output_data,
                         test_size=split_test_size,
                         random_state=split_random_state1)
    split_train_input, split_val_input, split_train_output, split_val_output = \
        train_test_split(split_train_val_input,
                         split_train_val_output,
                         test_size=split_val_size/(1-split_test_size),
                         random_state=split_random_state2)
    return split_train_input, split_val_input, split_test_input, split_train_output, split_val_output, split_test_output


def split_train(x_train, y_train, previous_size_reductions, new_test_size, pseudo_seed):
    """Reduce the training set in size.
    Parameters
    ----------
    x_train : np.ndarray
        Numpy array containing the input of all instances in the dataset.
    y_train : np.ndarray
        Numpy array containing the output of all instances in the dataset.
    previous_size_reductions : list[float]
        List of floats in [0,1] denoting the size of the previous reductions.
    new_test_size : float
        Float in (0,1] denoting the size of the new training size relative to the original size of the training set.
    pseudo_seed : list[int]
        List of integers denoting the pseudo random seeds used to choose the reductions of the training set.
    Returns
    ----------
    new_x_train : np.ndarray
        Numpy array containing part of the input of the instances in the dataset.
    new_y_train : np.ndarray
        Numpy array containing part of the output of the instances in the dataset.
    """
    new_x_train = x_train
    new_y_train = y_train
    previous_size = 1
    for i in range(0, len(previous_size_reductions)):
        new_x_train, _, new_y_train, _ = train_test_split(new_x_train,
                                                          new_y_train,
                                                          test_size=1-previous_size_reductions[i]/previous_size,
                                                          random_state=pseudo_seed[i])
        previous_size = previous_size_reductions[i]

    new_x_train, _, new_y_train, _ = train_test_split(new_x_train, new_y_train, test_size=(1-new_test_size)*previous_size, )

    return new_x_train, new_y_train


def make_dataset(x_data, y_data, batch_size):
    """Make a tf.data.Dataset of a given batch size. This function is used to convert the np.ndarrays of train, validation and test sets to
    another object that can be used by TensorFlow. The three sets are also divided into the different batches.
    Parameters
    ----------
    x_data : np.ndarray
        Numpy array containing the input of the instances in the dataset.
    y_data : np.ndarray
        Numpy array containing the output of the instances in the dataset.
    batch_size : int
        Integer denoting the batch size used.
    Returns
    ----------
    dataset : tf.data.Dataset
        Converted the arrays x_data, y_data to the right dtype for using in the training procedure.
    """

    x = tf.cast(x_data, dtype=tf.float32)
    y = tf.cast(y_data, dtype=tf.float32)
    y1 = y[:, 0]
    y2 = y[:, 1]
    t = tf.ones_like(y1)
    dataset = tf.data.Dataset.from_tensor_slices((x, y1, y2, t))
    dataset = dataset.batch(batch_size)
    return dataset

def t_loaders(seed, batch_size, number_weight_initializations, number_dataset_realizations, train_size,
                      validation_size, test_size):
    sd(seed)
    ran_min = round(random() * 10)  # take random lower bound on the pseudo random seeds used
    ran_max = round(random() * (10 ** 6))  # take random upper bound on the pseudo random seeds used

    weight_states = []  # list containing the pseudo random seeds used to initialize the network weights
    dataset_states1 = []  # list containing the pseudo random seeds used to divide the data in a test set and an other set
    dataset_states2 = []  # list containing the pseudo random seeds used to divide the other set into a train set and a validation set.
    dataset_states3 = []  # list containing the pseudo random seeds used to take a part of the training set.

    for i in range(0, number_weight_initializations):
        weight_states.append(randint(ran_min, ran_max))

    for i in range(0, number_dataset_realizations):
        dataset_states1.append(randint(ran_min, ran_max))
        dataset_states2.append(randint(ran_min, ran_max))

    train_size.sort(reverse=True)  # make sure that the largest element is first and the other elements are decreasing
    for i in range(0, len(train_size)):
        dataset_states3.append(randint(ran_min, ran_max))

    input_data, output_data = prep_data("examples/BAYESNET/data/", "Sample_Bias_correction_ucl_reduced", ".csv")
    x_train_complete, x_val, x_test, y_train_complete, y_val, y_test = split_train_test_val(input_data,
                                                                                            output_data,
                                                                                            test_size,
                                                                                            validation_size,
                                                                                            dataset_states1[0],
                                                                                            dataset_states2[0])

    n = len(train_size) - 1
    x_train, y_train = split_train(x_train_complete, y_train_complete, train_size[0:n], train_size[n], dataset_states3)

    # transform training data, validation data and test data to tf.data.Dataset
    D = make_dataset(x_train, y_train, batch_size)
    D_val = make_dataset(x_val, y_val, batch_size)
    D_test = make_dataset(x_test, y_test, batch_size)

    return D, D_val, D_test