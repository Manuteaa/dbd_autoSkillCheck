import dbd.datasets.datasetLoader as datasetLoader

import torchvision.transforms as tf
import torch

dataset_root = "tests/data"
dataset = datasetLoader._parse_dbd_datasetfolder(dataset_root)


def test_dbd_dataset_transforms():
    dbd_dataset = datasetLoader.DBD_dataset(dataset, tf.Compose([tf.ConvertImageDtype(torch.float32)]))

    x, y = dbd_dataset[0]
    assert y.ndim == 0
    assert x.ndim == 3
    assert x.dtype == torch.float32


def test_dbd_dataset_transforms2():
    transforms = tf.Compose([tf.ConvertImageDtype(torch.float32), tf.CenterCrop(10)])

    dbd_dataset = datasetLoader.DBD_dataset(dataset, transforms)
    x, y = dbd_dataset[0]
    assert x.ndim == 3
    assert x.dtype == torch.float32
    assert x.shape == (3, 10, 10)


def test_dbd_dataset_sampler():
    dbd_dataset = datasetLoader.DBD_dataset(dataset, tf.Compose([]))
    sampler = dbd_dataset._get_sampler()
    assert sampler.num_samples == len(dbd_dataset)

    w = sampler.weights

    # compute sum of weights for each class without for loop
    w_sum = torch.bincount(dbd_dataset.targets, weights=w)
    assert torch.allclose(w_sum, torch.ones_like(w_sum))

