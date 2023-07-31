import random
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import torch

from pathlib import Path


_DATA_ROOT = Path(__file__).parent
batch_size = 10

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def get_probability(d1, d2, continuous_hearing=True):
    e = d1
    b = d2 - 8
    e_p = 0.35 * e
    if continuous_hearing:
        # Ground truth neighbour distribution
        x_loc = tf.random.normal(d1.shape, 6, 3)
        y_loc = tf.random.normal(d1.shape, 3, 3)

        condition = tf.where(x_loc ** 2 + y_loc ** 2 < 100, 1., 0.)
        hears = condition
        calling_prob = hears * (e * e_p + b * 0.9 - e * b * 0.9 * e_p)
        sampled = tf.where(tf.random.uniform(calling_prob.shape) < calling_prob, 1., 0.)
        return sampled
        # return calling_prob
    return 0.9 * (e * e_p + b * 0.9 - e * b * 0.9 * e_p)

def get_simple_probs(d1, d2, continuous_hearing=True):
    e = d1
    b = d2 - 8

    if continuous_hearing:
        x_loc = tf.random.normal([1], 6, 3)
        y_loc = tf.random.normal([1], 3, 3)
        condition = tf.where(x_loc ** 2 + y_loc ** 2 < 100, 1., 0.)
        hears = tf.reduce_mean(condition)
        return hears * (e * 0.7 + b * 0.9 - e * b * 0.9 * 0.7)
    return 0.9 * (e * 0.7 + b * 0.9 - e * b * 0.9 * 0.7)


def create_dataloaders(dataset_name, restrictions, seed=7, simple=True, continuous_hearing=True, batch_size=10,
                       direct=False):
    if dataset_name == "val":
        dataset = datasets["test"]
    elif dataset_name == "test":
        dataset = datasets["test"]
    else:
        dataset = datasets[dataset_name]
    image_indices = list(range(len(dataset)))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(image_indices)

    dataset_iter = iter(image_indices)
    data = []
    burglary_data = []
    earthquake_data = []
    try:
        while dataset_iter:
            next_id = next(dataset_iter)
            if dataset[next_id][1] in restrictions[0]:
                earthquake_data.append(next_id)
            elif dataset[next_id][1] in restrictions[1]:
                burglary_data.append(next_id)
    except StopIteration:
        pass

    for i in range(min(len(burglary_data), len(earthquake_data))):
        data.append(
            [
                [earthquake_data[i]], [burglary_data[i]]
            ]
        )

    max = 1000
    primordial_dl = []
    I1_batch_list = []
    I2_batch_list = []
    I3_batch_list = []
    D1_batch_list = []
    D2_batch_list = []
    for id, i in enumerate(data):
        if dataset_name == "val" and id > max:
            break
        elif dataset_name == 'test' and id < max:
            pass
        I1_batch_list.append(dataset[i[0][0]][0])
        I2_batch_list.append(dataset[i[1][0]][0])
        I3_batch_list.append(torch.ones([2]))
        D1_batch_list.append(torch.tensor(dataset[i[0][0]][1]))
        D2_batch_list.append(torch.tensor(dataset[i[1][0]][1]))
        if id % batch_size == batch_size - 1:
            I1_batch = torch.stack(I1_batch_list)
            I2_batch = torch.stack(I2_batch_list)
            I3_batch = torch.stack(I3_batch_list)
            D1_batch = torch.stack(D1_batch_list)
            D2_batch = torch.stack(D2_batch_list)

            I1 = tf.reshape(tf.constant(I1_batch.numpy()), [batch_size, 28, 28, 1])
            I2 = tf.reshape(tf.constant(I2_batch.numpy()), [batch_size, 28, 28, 1])
            I3 = tf.constant(I3_batch.numpy())
            d1 = tf.constant(D1_batch.numpy(), dtype=tf.float32)
            d2 = tf.constant(D2_batch.numpy(), dtype=tf.float32)

            if simple:
                expected_prob = get_simple_probs(d1, d2, continuous_hearing=continuous_hearing)
            else:
                expected_prob = get_probability(d1, d2, continuous_hearing=continuous_hearing)
            if direct:
                primordial_dl.append([I1, I2, d1, d2])
            else:
                primordial_dl.append(
                    [[I1, I2, I3], expected_prob])

            I1_batch_list = []
            I2_batch_list = []
            I3_batch_list = []
            D1_batch_list = []
            D2_batch_list = []

    return primordial_dl
