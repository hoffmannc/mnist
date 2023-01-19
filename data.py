import os

import numpy as np
import torch


def mnist(path="data/"):
    l = sorted(os.listdir(path))
    trainfile = [i for i in l if i.startswith("train")]
    testfile = [i for i in l if i.startswith("test")]

    train_images, test_images = np.empty((0, 28, 28)), np.empty((0, 28, 28))
    train_labels, test_labels = np.empty([0]), np.empty([0])

    for file in trainfile:
        data = np.load(path + file)
        train_images = np.concatenate([train_images, data["images"]])
        train_labels = np.concatenate([train_labels, data["labels"]])

    for file in testfile:
        data = np.load(path + file)
        test_images = np.concatenate([test_images, data["images"]])
        test_labels = np.concatenate([test_labels, data["labels"]])

    train_images = np.expand_dims(train_images, 1)
    test_images = np.expand_dims(test_images, 1)

    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).float()
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).float()

    return (train_images, train_labels), (test_images, test_labels)
