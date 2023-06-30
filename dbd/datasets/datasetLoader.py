import os.path
import torch
import numpy as np
from glob import glob
import tqdm

import torchvision.io
from PIL import Image
import math

from dbd.datasets.transforms import get_training_transforms, get_validation_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


class DBD_dataset(Dataset):
    """
    Dataset class for DBD dataset
    - Handles caching of images
    - Handles sampler to deal with class imbalance
    """

    def __init__(self, dataset, transform):
        """
        :param dataset: numpy array of [image_path, label]
        :param transform: torchvision transforms. Either whole transform or [caching, not_caching] transforms
        """

        self.images_path = dataset[:, 0]
        self.targets = torch.tensor(dataset[:, 1].astype(np.int64), dtype=torch.int64)

        self.transform_caching = transform[0] if isinstance(transform, tuple) else transform
        self.transform_no_caching = transform[1] if isinstance(transform, tuple) else Compose([])

        self.images = None  # cached images

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.images is None:
            image = self.get_image_from_path(idx)
        else:
            image = self.images[idx]

        image = self.transform_no_caching(image)
        target = self.targets[idx]
        return image, target

    def _get_class_weights(self):
        count_classes = torch.unique(self.targets, return_counts=True)[1]
        w_mapping = 1.0 / count_classes  # all classes have equal chance to be sampled
        return w_mapping

    def _get_sampler(self, seed=42):
        generator_torch = torch.Generator().manual_seed(seed)
        w = self._get_class_weights()
        w = w[self.targets]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=generator_torch)
        return sampler

    def prefetch_images(self, batch_size=32, num_workers=8):
        transform_caching = self.transform_no_caching  # deep copy
        self.transform_no_caching = Compose([])  # Do not apply transform_no_caching when prefetching

        fetcher = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        images = []
        for _, batch in tqdm.tqdm(enumerate(fetcher), total=len(fetcher), desc="Prefetching data"):
            x, y = batch
            images.append(x)

        images = torch.cat(images, dim=0)
        self.images = images
        self.transform_no_caching = transform_caching

    def get_image_from_path(self, idx):
        image = self.images_path[idx]
        image = read_image(image, mode=torchvision.io.ImageReadMode.RGB)
        image = self.transform_caching(image)
        return image

    def get_dataloader(self, batch_size=32, num_workers=0, use_balanced_sampler=False):
        sampler = self._get_sampler() if use_balanced_sampler else None
        dataloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        return dataloader


def _parse_dbd_datasetfolder(root_dataset_path):
    folders = os.scandir(root_dataset_path)
    images_all = []
    targets_all = []

    for folder in folders:
        name, path = folder.name, folder.path
        if name.isdigit():
            print("Parsing folder " + name)
        else:
            print("Skipping folder " + name)
            continue

        images = glob(os.path.join(path, "*.*")) + glob(os.path.join(path, "*", "*.*"))

        images_all += images
        targets_all += [name] * len(images)

    dataset = np.stack([images_all, targets_all], axis=-1)
    return dataset


def get_dataloaders(root_dataset_path, batch_size=32, seed=42, num_workers=0, cache=False):
    assert os.path.exists(root_dataset_path)

    # Parse dataset
    dataset = _parse_dbd_datasetfolder(root_dataset_path)

    # Shuffle dataset and split into a training set and a validation set
    generator = np.random.default_rng(seed)
    generator.shuffle(dataset)

    nb_samples_train = math.floor(0.8 * len(dataset))
    dataset_train, dataset_val = dataset[:nb_samples_train], dataset[nb_samples_train:]

    # Set dataloader
    train_transforms = get_training_transforms(decompose=True)
    dataset_train = DBD_dataset(dataset_train, train_transforms)
    if cache: dataset_train.prefetch_images()
    dataloader_train = dataset_train.get_dataloader(batch_size=batch_size, num_workers=num_workers, use_balanced_sampler=True)

    val_transforms = get_validation_transforms(decompose=True)
    dataset_val = DBD_dataset(dataset_val, val_transforms)
    if cache: dataset_val.prefetch_images()
    dataloader_val = dataset_val.get_dataloader(batch_size=batch_size, num_workers=num_workers, use_balanced_sampler=False)

    return dataloader_train, dataloader_val


if __name__ == '__main__':
    from dbd.datasets.transforms import MEAN, STD

    dataset_root = "dataset/"
    dataloader_train, dataloader_val = get_dataloaders(dataset_root)

    std = torch.tensor(STD, dtype=torch.float32).reshape((3, 1, 1))
    mean = torch.tensor(MEAN, dtype=torch.float32).reshape((3, 1, 1))
    for i, batch in enumerate(dataloader_train):
        x, y = batch
        x = x[0]  # take first sample
        x = x * std + mean  # un-normalization to [0, 1] with auto-broadcast
        x = x * 255.

        x = x.permute((1, 2, 0))  # channel last
        x = x.cpu().numpy().astype(np.uint8)

        img = Image.fromarray(x, "RGB")
        img.save(os.path.join(dataset_root, "{}.jpg".format(i)))
        print("Saving {}.jpg".format(i))
