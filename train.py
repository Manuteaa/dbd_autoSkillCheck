import numpy as np

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader

import Dbd_Model
import Dbd_DatasetLoader

import glob
import os

if __name__ == '__main__':
    ##########################################################
    # checkpoint = None
    checkpoint = "./lightning_logs/version_4/checkpoints/epoch=1-step=1316.ckpt"

    dataset_0 = ["E:/temp/dbd/0", "E:/temp/dbd/0bis"]
    dataset_1 = ["E:/temp/dbd/1", "E:/temp/dbd/1bis"]

    datasets_path = dataset_0 + dataset_1
    labels_idx = [0, 0] + [1, 1]
    ##########################################################

    filenames_labels_tuples = Dbd_DatasetLoader.get_filenames_labels(datasets_path, labels_idx)
    np.random.shuffle(filenames_labels_tuples)
    nb_training_samples = int(filenames_labels_tuples.shape[0] * 0.8)
    training_filenames_labels, validation_filenames_labels = filenames_labels_tuples[:nb_training_samples], filenames_labels_tuples[nb_training_samples:]

    # TODO
    dataset_training = Dbd_DatasetLoader.CustomDataset(datasets_path, labels_idx, transform=Dbd_DatasetLoader.transforms)

    training_data, validation_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    training_sampler = Dbd_DatasetLoader.get_randomSampler(training_data)
    training_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=64, pin_memory=True, num_workers=8)
    validation_dataloader = DataLoader(training_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)
    test = next(iter(training_dataloader))

    my_model = Dbd_Model.My_Model() if checkpoint is None else Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5)
    trainer.fit(model=my_model, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
