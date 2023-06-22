import os.path
import torch
import numpy as np
from glob import glob
from PIL import Image
import math

from dbd.datasets.transforms import get_training_transforms, get_validation_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

class DBD_dataset(Dataset):
    def __init__(self, dataset, transform):
        self.images_path = dataset[:, 0]
        self.targets = np.array(dataset[:, 1], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images_path[idx]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)

        target = self.targets[idx]  # conversion to tensor is done in default collate

        return image, target

    def _get_class_weights(self):
        count_classes = np.unique(self.targets, return_counts=True)[1]
        w_mapping = 1.0 / count_classes  # all classes have equal chance to be sampled
        return w_mapping

    def get_sampler(self, seed=42):
        generator_torch = torch.Generator().manual_seed(seed)
        w = self._get_class_weights()
        w = w[self.targets]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=generator_torch)
        return sampler

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

def get_dataloaders(root_dataset_path, batch_size=32, seed=42, num_workers=0):
    assert os.path.exists(root_dataset_path)

    # Parse dataset
    dataset = _parse_dbd_datasetfolder(root_dataset_path)

    # Shuffle dataset and split into a training set and a validation set
    generator = np.random.default_rng(seed)
    generator.shuffle(dataset)

    nb_samples_train = math.floor(0.8 * len(dataset))
    dataset_train, dataset_val = dataset[:nb_samples_train], dataset[nb_samples_train:]

    # Set datasets
    training_transforms = get_training_transforms()
    dataset_train = DBD_dataset(dataset_train, training_transforms)

    validation_transforms = get_validation_transforms()
    dataset_val = DBD_dataset(dataset_val, validation_transforms)

    # Set dataloaders
    dataloader_train = DataLoader(dataset_train, sampler=dataset_train.get_sampler(), batch_size=batch_size, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, sampler=dataset_val.get_sampler(), batch_size=batch_size, num_workers=num_workers)

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
