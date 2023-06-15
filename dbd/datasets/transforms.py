import torchvision.transforms as tf
import torch

def get_training_transforms():
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.RandomResizedCrop(224, scale=(0.8, 1.0)),
        tf.RandomVerticalFlip(0.2),
        tf.RandomHorizontalFlip(0.2),
        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        tf.ConvertImageDtype(torch.float32)
        # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms

def get_validation_transforms():
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.CenterCrop(224),
        tf.ConvertImageDtype(torch.float32)
        # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    dataset_root = "dataset/"
    transforms_train, transforms_valid = get_training_transforms(), get_validation_transforms()
    dataloader_train, _ = get_dataloaders(dataset_root, transforms_train, transforms_valid)

    for i, batch in enumerate(dataloader_train):
        x, y = batch
        x = x[0]
        x = x.permute((1, 2, 0))

        x = x * 255.
        x = x.cpu().numpy().astype(np.uint8)

        img = Image.fromarray(x, "RGB")
        img.save(os.path.join(dataset_root, "{}.jpg".format(i)))
        print("Saving {}.jpg".format(i))
