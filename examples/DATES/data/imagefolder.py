import re
import tensorflow as tf
import random
import numpy as np

from typing import Any
from torchvision import datasets

def image_regresstarget(path, digits):
    numbers = tf.constant([float(i) for i in re.findall(r'\d+', path)])
    x_locs = [(numbers[i + 3] + 14) / 200 for i in range(digits)]
    y_locs = [(numbers[i + 3 + digits] + 14) / 120 for i in range(digits)]
    return tf.stack([tf.stack([x_locs[i], y_locs[i]]) for i in range(digits)])

def image_target(path):
    return int(re.findall(r'\d+', path)[1])

class RegressionImageFolder(datasets.ImageFolder):

    def __init__(
        self, root: str, batch_size: int, digits: int, **kwargs: Any
    ) -> None:
        super().__init__(root, **kwargs)
        # self.transform = tf.keras.layers.Normalization(mean=0.5, variance=0.5)
        # self.transform.mean, self.transform.variance = 0.5, 0.5
        self.batch_size = batch_size
        paths, _ = zip(*self.imgs)
        self.regresstargets = [image_regresstarget(path, digits) for path in paths]
        self.targets = [image_target(path) for path in paths]
        self.samples = self.imgs = list(zip(paths, self.regresstargets, self.targets))

        rng = random.Random(7)
        c = list(zip(self.regresstargets, self.targets, self.samples))
        rng.shuffle(c)
        self.regresstargets, self.targets, self.samples = zip(*c)


        if batch_size is not None:
            L = len(self.samples)
            batchsamples = []
            batchregress = []
            batchtargets = []
            sample_batch = []
            regress_batch = []
            target_batch = []
            for i in range(L):
                if i % batch_size == 0 and i != 0:
                    batchsamples.append(sample_batch)
                    batchregress.append(regress_batch)
                    batchtargets.append(target_batch)
                    sample_batch = []
                    regress_batch = []
                    target_batch = []
                sample_batch.append(self.samples[i])
                regress_batch.append(self.regresstargets[i])
                target_batch.append(self.targets[i])
            self.samples = batchsamples
            self.regresstargets = batchregress
            self.targets = batchtargets


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, regress, target) where target is class_index of the target class.
        """
        paths = [x[0] for x in self.samples[index]]
        regress = [x[1] for x in self.samples[index]]
        target = tf.stack([x[2] for x in self.samples[index]])
        samples = []
        for path in paths:
            img = tf.constant(np.array(self.loader(path)))
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(img, dtype=tf.float32) / 255. 
            # img = self.transform.call(img)
            samples.append(img)
        samples = tf.stack(samples)
        regress = tf.stack(regress)
        return samples, regress, target
