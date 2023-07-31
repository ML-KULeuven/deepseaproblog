import random
import tensorflow as tf
import pickle

from pathlib import Path
from examples.DATES.data.imagefolder import RegressionImageFolder


def load_dataset(batch_size, digits=4):
    _DATA_ROOT = Path(__file__).parent

    if digits == 2:
        datasets = {
            "train": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_2_digits/train'),
            "val": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_2_digits/val'),
            "test": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_2_digits/test')
        }
    else:
        datasets = {
            "train": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_4_digits_same/train'),
            "val": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_4_digits_same/val'),
            "test": RegressionImageFolder(batch_size=batch_size, digits=digits, root='examples/DATES/data/year_4_digits_same/test')
        }

    return datasets

def create_dataloader(dataset_name, digits, batch_size, cur="", data_size=None):
    data = load_dataset(batch_size, digits=digits)[dataset_name]
    image_idx = list(range(len(data)))

    rng = random.Random(7)
    rng.shuffle(image_idx)

    dataloader = []
    if data_size is not None:
        size = data_size // batch_size
    else:
        size = len(data)

    for id in image_idx[:size]:
        image, regress, year = data[id]
        image = image * 2 - 1

        coords = []
        for i in range(digits):
            coords.append(regress[:, i, :])
        if cur == "":
            dataloader.append([[image, year], tf.constant([1.] * batch_size, dtype=tf.float32)])
        elif cur == "_regress":
            dataloader.append([[image, year] + coords, tf.constant([1.] * batch_size, dtype=tf.float32)])
        elif cur == "_cur":
            supervision = []
            for i in range(digits):
                supervision.insert(0, year - year // 10 * 10)
                year = year // 10
            dataloader.append([[image] + supervision, tf.constant([1.] * batch_size, dtype=tf.float32)])
        else:
            supervision = []
            for i in range(digits):
                supervision.insert(0, year - year // 10 * 10)
                year = year // 10
            dataloader.append([[image] + supervision + coords, tf.constant([1.] * batch_size, dtype=tf.float32)])
    return dataloader
