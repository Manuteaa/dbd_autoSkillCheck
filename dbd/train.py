import pytorch_lightning as pl
import glob
import os

from dbd.datasets.datasetLoader import get_dataloaders
from dbd.networks.model import Model


if __name__ == '__main__':
    ##########################################################
    checkpoint = "./lightning_logs/version_2/checkpoints"
    dataset_root = "dataset/"

    ##########################################################
    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]
    assert os.path.isfile(checkpoint)

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, num_workers=0, cache=True)

    # Model
    model = Model(lr=5e-5)
    model.load_from_checkpoint(checkpoint)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=500)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # tensorboard --logdir=lightning_logs/

