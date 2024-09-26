import torchvision.transforms.v2 as tf
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_training_transforms():
    transforms = tf.Compose([
        # Random rotation to augment dataset
        tf.RandomRotation(180),

        # Random "zoom" to gain skill check position and scale robustness
        tf.CenterCrop(224),
        tf.RandomResizedCrop(224, scale=(0.8, 1.0)),

        # Random color filters
        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

        tf.ToDtype(torch.float32, scale=True),
        tf.Normalize(mean=MEAN, std=STD)
    ])

    return transforms


def get_validation_transforms():
    transforms = tf.Compose([
        tf.CenterCrop(224),
        tf.ToDtype(torch.float32, scale=True),
        tf.Normalize(mean=MEAN, std=STD)
    ])
    return transforms
