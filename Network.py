import os.path
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torchmetrics

import torch
from torch.utils.data import Dataset
import torchvision.io
import torchvision.models as models
import torchvision.transforms as tf

class CustomDataset(Dataset):
    def __init__(self, directories, directories_class_id, transform=None):
        assert len(directories) == len(directories_class_id)
        self.transform = transform

        self.filenames = []
        self.labels = []
        for i in range(len(directories)):
            directory = directories[i]
            label = directories_class_id[i]

            filenames = glob(os.path.join(directory, "*.png")) + glob(os.path.join(directory, "*.jpg"))
            labels = [label] * len(filenames)

            self.filenames += filenames
            self.labels += labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.filenames[idx], torchvision.io.ImageReadMode.RGB)
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def getSampler(self):
        # Deal with imbalanced dataset, using a WeightedRandomSampler
        labels = np.array(self.labels)
        nb_labels_1 = np.count_nonzero(labels)  # Only for binary classification!
        nb_labels_0 = len(self.labels) - nb_labels_1
        probs = [1 / nb_labels_0, 1 / nb_labels_1]
        w = [probs[0]] * nb_labels_0 + [probs[1]] * nb_labels_1
        sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
        return sampler


transforms_test = tf.Compose([
    tf.CenterCrop(224),
    tf.ConvertImageDtype(torch.float32)
])

transforms = tf.Compose([
    tf.RandomResizedCrop(224, scale=(0.8, 1.0)),
    tf.RandomVerticalFlip(0.2),
    tf.RandomHorizontalFlip(0.2),
    tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    tf.ConvertImageDtype(torch.float32)
    # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.encoder.eval()

        # freeze params
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=2)

    def build_encoder(self):
        weights = models.MobileNet_V2_Weights.DEFAULT
        return models.mobilenet_v2(weights=weights)

    def build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 2),
            torch.nn.Softmax()
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        z = self.encoder(x)
        pred = self.decoder(z)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.accuracy(pred, y)
        self.recall(pred, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log('train_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_recall', self.recall, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer
