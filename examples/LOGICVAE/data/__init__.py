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


def batch_joint_subtraction(d1, d2):
    return tf.subtract(d1, d2)

def create_dataloaders(dataset_name, seed=7):
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

    try:
        while dataset_iter:
            next_id = next(dataset_iter)
            data.append(next_id)
    except StopIteration:
        pass

    primordial_dl = []
    I_batch_list = []
    for id, i in enumerate(data):
        if dataset_name == 'val' and id > 100:
            break
        I_batch_list.append(dataset[i][0])
        if id % batch_size == 9:
            I_batch = torch.stack(I_batch_list)

            I = tf.reshape(tf.constant(I_batch.numpy()), [batch_size, 28, 28, 1])
            primordial_dl.append([I, 1.0])

            I_batch_list = []

    return primordial_dl

def create_simple_dataloaders(dataset_name, seed=7, digits=1, curriculum=False, data_size=1000, batch_size=10):
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
    try:
        while dataset_iter:
            data.append(
                [
                    [next(dataset_iter) for _ in range(digits)]
                    for _ in range(2)
                ]
            )
    except StopIteration:
        pass

    primordial_dl = []
    I1_batch_list = []
    I2_batch_list = []
    D1_batch_list = []
    D2_batch_list = []
    for id, i in enumerate(data):
        if dataset_name == 'val' and id > data_size:
            break
        if curriculum and id > data_size:
            break
        I1_sublist = dataset[i[0][0]][0]
        I2_sublist = dataset[i[1][0]][0]
        D1_sublist = torch.tensor(dataset[i[0][0]][1])
        D2_sublist = torch.tensor(dataset[i[1][0]][1])

        I1_batch_list.append(I1_sublist)
        I2_batch_list.append(I2_sublist)
        D1_batch_list.append(D1_sublist)
        D2_batch_list.append(D2_sublist)
        if id % batch_size == batch_size - 1:
            I1_batch = torch.stack(I1_batch_list)
            I2_batch = torch.stack(I2_batch_list)
            D1_batch = torch.stack(D1_batch_list)
            D2_batch = torch.stack(D2_batch_list)

            I1 = tf.reshape(tf.constant(I1_batch.numpy()), [batch_size, 28, 28, 1])
            I2 = tf.reshape(tf.constant(I2_batch.numpy()), [batch_size, 28, 28, 1])
            d1 = tf.constant(D1_batch.numpy(), dtype=tf.float32)
            d2 = tf.constant(D2_batch.numpy(), dtype=tf.float32)

            if curriculum:
                primordial_dl.append(
                    [[I1, I2, d1, d2], tf.constant([1.] * len(d1), dtype=tf.float32)])
            else:
                expected_out = batch_joint_subtraction(d1, d2)
                primordial_dl.append(
                    [[I1, I2, expected_out], tf.constant([1.] * len(d1), dtype=tf.float32)])

            I1_batch_list = []
            I2_batch_list = []
            D1_batch_list = []
            D2_batch_list = []

    return primordial_dl