import os.path
import torch

from torch.utils.data import random_split, DataLoader
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

    #Set dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, generator=generator)  # TODO: use WeightedRandomSampler to deal with class imbalance?
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)

    return dataloader_train, dataloader_val

