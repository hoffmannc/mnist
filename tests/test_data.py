import torch

from tests import _PATH_DATA

from data import mnist


def test_data():
    train_data, test_data = mnist(path="data_test/")
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    assert train_images.size()[2:4] == torch.Size([28, 28]), "Unexpected image shape"
    assert test_images.size()[2:4] == torch.Size([28, 28]), "Unexpected image shape"
