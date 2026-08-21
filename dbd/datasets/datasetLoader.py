"""Dataset discovery and PyTorch data loader helpers."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from dbd.datasets.transforms import get_training_transforms, get_validation_transforms


class DBDDataset(Dataset):
    """Load labeled Dead by Daylight images and apply a transform."""

    def __init__(self, samples: np.ndarray, transform: Callable) -> None:
        """Initialize the dataset from ``(image path, label)`` pairs."""
        self.image_paths = samples[:, 0]
        self.targets = torch.as_tensor(samples[:, 1].astype(np.int64), dtype=torch.int64)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Decode and transform one image with its target."""
        image = decode_image(str(self.image_paths[index]), mode=ImageReadMode.RGB)
        return self.transform(image), self.targets[index]


def parse_dbd_datasetfolder(root_dataset_path: str | Path) -> np.ndarray:
    """Return sorted ``(image path, label)`` pairs from numeric class folders."""
    root = Path(root_dataset_path)
    samples = []
    for class_folder in sorted(root.iterdir(), key=lambda path: path.name):
        if not class_folder.is_dir() or not class_folder.name.isdigit():
            continue
        image_paths = sorted(class_folder.glob("*.*"), key=lambda path: path.name)
        samples.extend((str(image_path), class_folder.name) for image_path in image_paths if image_path.is_file())

    return np.asarray(samples, dtype=str).reshape(-1, 2)


def get_dataloaders(
    root_dataset_path: str | Path,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    ratio_train: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic training and validation data loaders."""
    samples = parse_dbd_datasetfolder(root_dataset_path)
    np.random.default_rng(seed).shuffle(samples)
    split_index = int(ratio_train * len(samples))
    training_samples, validation_samples = samples[:split_index], samples[split_index:]

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": True,
    }
    training_loader = DataLoader(
        DBDDataset(training_samples, get_training_transforms()),
        **loader_options,
    )
    validation_loader = DataLoader(
        DBDDataset(validation_samples, get_validation_transforms()),
        **loader_options,
    )
    return training_loader, validation_loader
