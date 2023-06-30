import dbd.datasets.datasetLoader as datasetLoader

import torchvision.transforms as tf
import torch

dataset_root = "tests/data"
dataset = datasetLoader._parse_dbd_datasetfolder(dataset_root)


def test_dbd_dataset_no_transforms():
    dbd_dataset = datasetLoader.DBD_dataset(dataset, tf.Compose([]))
    x = dbd_dataset.get_image_from_path(0)
    assert x.ndim == 3
    assert x.dtype == torch.uint8

    x2, y2 = dbd_dataset[0]
    assert y2.ndim == 0
    assert x2.ndim == 3
    assert x2.dtype == torch.uint8
    assert torch.allclose(x, x2)


def test_dbd_dataset_single_transforms():
    dbd_dataset = datasetLoader.DBD_dataset(dataset, tf.Compose([tf.ConvertImageDtype(torch.float32)]))
    x = dbd_dataset.get_image_from_path(0)
    assert x.ndim == 3
    assert x.dtype == torch.float32

    x2, y2 = dbd_dataset[0]
    assert y2.ndim == 0
    assert x2.ndim == 3
    assert x2.dtype == torch.float32
    assert torch.allclose(x, x2)


def test_dbd_dataset_different_transforms():
    cached_transform = tf.Compose([tf.ConvertImageDtype(torch.float32)])
    not_cached_transform = tf.Compose([tf.CenterCrop(10)])

    dbd_dataset = datasetLoader.DBD_dataset(dataset, (cached_transform, not_cached_transform))
    x = dbd_dataset.get_image_from_path(0)
    assert x.ndim == 3
    assert x.dtype == torch.float32

    x2, y2 = dbd_dataset[0]
    assert x2.ndim == 3
    assert x2.dtype == torch.float32
    assert x2.shape == (3, 10, 10)


def test_dbd_dataset_sampler():
    dbd_dataset = datasetLoader.DBD_dataset(dataset, tf.Compose([]))
    sampler = dbd_dataset._get_sampler()
    assert sampler.num_samples == len(dbd_dataset)

    w = sampler.weights

    # counts = torch.unique(dbd_dataset.targets, return_counts=True)[1]
    # compute sum of weights for each class
    # for class_idx in range(len(counts)):
    #     w_sum = torch.sum(w[dbd_dataset.targets == class_idx])
    #     assert torch.allclose(w_sum, torch.ones_like(w_sum))

    # compute sum of weights for each class without for loop
    w_sum = torch.bincount(dbd_dataset.targets, weights=w)
    assert torch.allclose(w_sum, torch.ones_like(w_sum))


def test_dbd_fetching():
    cached_transform = tf.Compose([tf.ConvertImageDtype(torch.float32), tf.CenterCrop(20)])
    not_cached_transform = tf.Compose([tf.CenterCrop(10)])
    dbd_dataset = datasetLoader.DBD_dataset(dataset, (cached_transform, not_cached_transform))

    assert dbd_dataset.images is None

    dbd_dataset.prefetch_images(num_workers=0)

    assert dbd_dataset.images is not None
    assert len(dbd_dataset.images) == len(dbd_dataset)

    x = dbd_dataset.images[0]
    assert x.ndim == 3
    assert x.dtype == torch.float32
    assert x.shape == (3, 20, 20)

    x2 = dbd_dataset.get_image_from_path(0)
    assert torch.allclose(x, x2)

    x3, y3 = dbd_dataset[0]
    assert x3.shape == (3, 10, 10)
    assert x3.dtype == torch.float32


