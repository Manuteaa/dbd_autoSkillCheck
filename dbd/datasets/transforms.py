import torchvision.transforms as tf
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_training_transforms(decompose=False):
    transforms_cache = tf.Compose([
        tf.ConvertImageDtype(torch.float32),
    ])

    transforms_no_cache = tf.Compose([
        tf.RandomRotation(180),
        tf.RandomResizedCrop(224, scale=(0.8, 1.0)),
        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        tf.Normalize(mean=MEAN, std=STD)
        ])

    if decompose:
        return transforms_cache, transforms_no_cache

    return tf.Compose([transforms_cache, transforms_no_cache])

def get_validation_transforms(decompose=False):
    transforms_cache = tf.Compose([
        tf.ConvertImageDtype(torch.float32),
        tf.CenterCrop(224),
        tf.Normalize(mean=MEAN, std=STD)
    ])

    transforms_no_cache = tf.Compose([])

    if decompose:
        return transforms_cache, transforms_no_cache

    return tf.Compose([transforms_cache, transforms_no_cache])
