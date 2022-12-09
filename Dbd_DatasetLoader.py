import os.path
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.io
import torchvision.transforms as tf


def get_randomSampler(labels):
    # Deal with imbalanced dataset, using a WeightedRandomSampler
    labels = np.array(labels)
    nb_labels_1 = np.count_nonzero(labels)  # Only for binary classification!
    nb_labels_0 = len(labels) - nb_labels_1
    probs = [1 / nb_labels_0, 1 / nb_labels_1]
    w = [probs[0]] * nb_labels_0 + [probs[1]] * nb_labels_1
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    return sampler


class CustomDataset(Dataset):
    def __init__(self, directories, directories_class_id, transform=None):
        assert len(directories) == len(directories_class_id)
        self.transform = transform

        self.filenames = []
        self.labels = []
        for i in range(len(directories)):
            directory = directories[i]
            label = directories_class_id[i]

            filenames = glob(os.path.join(directory, "*.png")) + glob(os.path.join(directory, "*.jpg"))
            labels = [label] * len(filenames)

            self.filenames += filenames
            self.labels += labels

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