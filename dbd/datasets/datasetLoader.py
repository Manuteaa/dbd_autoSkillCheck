import os.path
import torch
import numpy as np
from glob import glob

import torchvision.io
import math

from dbd.datasets.transforms import get_training_transforms, get_validation_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.io import read_image


class DBD_dataset(Dataset):
    """
    Dataset class for DBD dataset
    - Handles custom sampler to deal with class imbalance
    """

    def __init__(self, dataset, transforms):
        """
        :param dataset: numpy array of {image_path, label}
        :param transforms: torchvision transforms
        """

        self.images_path = dataset[:, 0]
        self.targets = torch.tensor(dataset[:, 1].astype(np.int64), dtype=torch.int64)
        self.transforms = transforms

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.get_image_from_path(idx)
        image = self.transforms(image)

        target = self.targets[idx]
        return image, target

    def _get_class_weights(self):
        count_classes = torch.bincount(self.targets)
        w_mapping = 1.0 / count_classes  # all classes have equal chance to be sampled
        return w_mapping

    def _get_sampler(self, seed=42):
        generator_torch = torch.Generator().manual_seed(seed)
        w = self._get_class_weights()
        w = w[self.targets]

        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=generator_torch)
        return sampler

    def get_image_from_path(self, idx):
        image = self.images_path[idx]
        image = read_image(image, mode=torchvision.io.ImageReadMode.RGB)
        return image

    def get_dataloader(self, batch_size=32, num_workers=0, use_balanced_sampler=False):
        sampler = self._get_sampler() if use_balanced_sampler else None
        dataloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=sampler, persistent_workers=True, pin_memory=True)
        return dataloader


def _parse_dbd_datasetfolder(root_dataset_path):
    """
    Get dataset as list of pairs {image path, label} in numpy array format
    Args:
        root_dataset_path:

    Returns: numpy array with shape (nb_images, 2), data type is str

    """
    folders = os.scandir(root_dataset_path)
    images_all = []
    targets_all = []

    for folder in folders:
        name, path = folder.name, folder.path
        if not name.isdigit():
            print("Skipping folder " + name)
            continue

        images = glob(os.path.join(path, "*.*"))
        print("Parsing folder {} : {} images found".format(name, len(images)))

        images_all += images
        targets_all += [name] * len(images)

    dataset = np.stack([images_all, targets_all], axis=-1)
    return dataset


def get_dataloaders(root_dataset_path, batch_size=32, seed=42, num_workers=0):
    """  Get training and validation data loaders
    Args:
        root_dataset_path: Root dataset path, containing folders with name corresponding to associated class
        batch_size: batch size
        seed: seed to init random generators
        num_workers: data loader num workers

    """
    assert os.path.exists(root_dataset_path)

    # Parse dataset
    dataset = _parse_dbd_datasetfolder(root_dataset_path)  # shape is (nb_images, 2)

    # Shuffle dataset and split into a training set and a validation set
    generator = np.random.default_rng(seed)
    generator.shuffle(dataset)

    nb_samples_train = math.floor(0.8 * len(dataset))
    dataset_train, dataset_val = dataset[:nb_samples_train], dataset[nb_samples_train:]

    # Set data loaders
    train_transforms = get_training_transforms()
    dataset_train = DBD_dataset(dataset_train, train_transforms)
    dataloader_train = dataset_train.get_dataloader(batch_size=batch_size, num_workers=num_workers, use_balanced_sampler=True)

    val_transforms = get_validation_transforms()
    dataset_val = DBD_dataset(dataset_val, val_transforms)
    dataloader_val = dataset_val.get_dataloader(batch_size=batch_size, num_workers=num_workers, use_balanced_sampler=False)

    return dataloader_train, dataloader_val


if __name__ == '__main__':
    from dbd.datasets.transforms import MEAN, STD
    import cv2

    dataset_root = "dataset/"
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, batch_size=1, num_workers=1)
    # dataloader_train, dataloader_val = get_dataloaders(dataset_root, batch_size=32, num_workers=1)

    std = torch.tensor(STD, dtype=torch.float32).reshape((3, 1, 1))
    mean = torch.tensor(MEAN, dtype=torch.float32).reshape((3, 1, 1))

    batch = next(iter(dataloader_train))
    x, y = batch

    # for batch in dataloader_train:
    #     x, y = batch
    #     print(torch.bincount(y))

    for i, batch in enumerate(dataloader_train):
        x, y = batch
        x = x[0]  # take first sample
        x = x * std + mean  # un-normalization to [0, 1] with auto-broadcast
        x = x * 255.

        x = x.permute((1, 2, 0))  # channel last : (3, 224, 224) --> (224, 224, 3)
        x = x.cpu().numpy().astype(np.uint8)

        img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        category = str(y.cpu().numpy()[0])
        cv2.imshow(category, img)
        cv2.moveWindow(category, 200, 200)
        cv2.waitKey()
