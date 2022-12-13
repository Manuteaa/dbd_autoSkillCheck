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
    checkpoint = None
    # checkpoint = "./lightning_logs/version_4/checkpoints/epoch=1-step=1316.ckpt"

    datasets_path = ["E:/temp/dbd/0", "E:/temp/dbd/0bis", "E:/temp/dbd/1", "E:/temp/dbd/1bis", "E:/temp/dbd/2"]
    labels_idx = [0, 0, 1, 1, 2]
    ##########################################################

    filenames_labels_tuples = Dbd_DatasetLoader.get_filenames_labels(datasets_path, labels_idx)
    np.random.shuffle(filenames_labels_tuples)
    nb_training_samples = int(filenames_labels_tuples.shape[0] * 0.8)
    training_data, validation_data = filenames_labels_tuples[:nb_training_samples], filenames_labels_tuples[nb_training_samples:]

    training_dataset = Dbd_DatasetLoader.CustomDataset(data_tuples=training_data, transform=Dbd_DatasetLoader.transforms)
    validation_dataset = Dbd_DatasetLoader.CustomDataset(data_tuples=validation_data, transform=Dbd_DatasetLoader.transforms_test)

    training_sampler = Dbd_DatasetLoader.get_randomSampler(training_dataset)
    training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=64, pin_memory=True, num_workers=1)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=1)
    # test = next(iter(training_dataloader))

    my_model = Dbd_Model.My_Model() if checkpoint is None else Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5)
    trainer.fit(model=my_model, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
