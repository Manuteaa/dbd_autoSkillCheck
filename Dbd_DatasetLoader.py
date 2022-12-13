import os.path
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.io
import torchvision.transforms as tf


def get_filenames_labels(datasets_path, labels_idx):
    assert len(datasets_path) == len(labels_idx)

    all_filenames = []
    all_labels = []
    for i in range(len(datasets_path)):
        directory = datasets_path[i]
        label = labels_idx[i]

        filenames = glob(os.path.join(directory, "*.png")) + glob(os.path.join(directory, "*.jpg"))
        labels = [label] * len(filenames)

        all_filenames += filenames
        all_labels += labels

    result_tuples = np.stack([all_filenames, all_labels], -1)
    return result_tuples


def get_randomSampler(dataset):
    # Deal with imbalanced dataset, using a WeightedRandomSampler
    label_max = torch.max(dataset.labels)
    nb_labels = dataset.labels.shape[0]
    prob_labels = [torch.count_nonzero(dataset.labels == i) / nb_labels for i in range(label_max+1)]
    w = [1.0 - prob_labels[label] for label in dataset.labels]
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    return sampler


class CustomDataset(Dataset):
    def __init__(self, filenames=None, labels=None, data_tuples=None, transform=None):
        if data_tuples is not None:
            filenames, labels = np.transpose(data_tuples, (1, 0))
            labels = labels.astype(np.int64)

        assert filenames is not None and labels is not None and len(filenames) == len(labels)

        self.filenames = filenames
        self.labels = torch.tensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.filenames[idx], torchvision.io.ImageReadMode.RGB)
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


transforms_test = tf.Compose([
    tf.CenterCrop(224),
    tf.ConvertImageDtype(torch.float32)
])

transforms = tf.Compose([
    tf.RandomResizedCrop(224, scale=(0.8, 1.0)),
    tf.RandomVerticalFlip(0.2),
    tf.RandomHorizontalFlip(0.2),
    tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    tf.ConvertImageDtype(torch.float32)
    # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])