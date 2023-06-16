import os.path
import torch
import numpy as np

from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder


def get_dataloaders(root, transforms_train, transforms_val, seed=42):
    assert os.path.exists(root)

    # Use ImageDataset for integrated (image, class_target) parsing following dataset structure
    full_dataset = ImageFolder(root)

    # Split training and validation
    generator = torch.Generator().manual_seed(seed)
    dataset_train, dataset_val = random_split(full_dataset, [0.8, 0.2], generator)

    # Set correct transform (note that setting transforms (with an 's') is not necessary)
    dataset_train.dataset.transform = transforms_train
    dataset_val.dataset.transform = transforms_val

    # Get sampler
    count_classes = np.unique(full_dataset.targets, return_counts=True)[1]
    nb_classes = count_classes.size

    w_mapping = 1.0 / count_classes  # all classes have equal chance to be sampled. Note that it does not sum up to one
    w_mapping[0] = w_mapping[0] * (nb_classes - 1)  # we want p(sampling class 0)=0.5 and p(sampling class not 0)=0.5
    w = w_mapping[full_dataset.targets]

    w_train = w[dataset_train.indices]  # get associated subset of weight
    sampler_train = WeightedRandomSampler(w_train, num_samples=32, replacement=True, generator=generator)

    w_val = w[dataset_val.indices]  # get associated subset of weight
    sampler_val = WeightedRandomSampler(w_val, num_samples=32, replacement=True, generator=generator)

    #Set dataloaders
    dataloader_train = DataLoader(dataset_train, sampler=sampler_train)
    dataloader_val = DataLoader(dataset_val, sampler=sampler_val)

    return dataloader_train, dataloader_val

