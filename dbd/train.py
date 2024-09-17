import glob
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

from dbd.datasets.datasetLoader import get_dataloaders
from dbd.networks.model import Model


# torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    ##########################################################
    checkpoint = "./lightning_logs/version_1/checkpoints"
    dataset_root = "dataset/"

    ##########################################################
    # checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, num_workers=8)

    # Model
    model = Model(lr=1e-4)
    # model = Model.load_from_checkpoint(checkpoint, strict=True, lr=1e-4)

    # Print model summary
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    # Compile the model
    # model = torch.compile(model)

    valid = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    valid.validate(model=model, dataloaders=dataloader_val)

    # Training
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="loss/val")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=100, num_sanity_val_steps=0, precision="16-mixed", callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # tensorboard --logdir=lightning_logs/
