import torchvision.transforms as tf

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_training_transforms():
    transforms = tf.Compose([
        tf.RandomResizedCrop(224, scale=(0.8, 1.0)),
        tf.RandomVerticalFlip(0.2),
        tf.RandomHorizontalFlip(0.2),
        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        tf.ToTensor(),
        tf.Normalize(mean=MEAN, std=STD)
    ])
    return transforms

def get_validation_transforms():
    transforms = tf.Compose([
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize(mean=MEAN, std=STD)
    ])
    return transforms
