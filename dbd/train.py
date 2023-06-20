import pytorch_lightning as pl

from dbd.datasets.datasetLoader import get_dataloaders
from dbd.networks.model import Model


if __name__ == '__main__':
    ##########################################################
    # checkpoint = "./lightning_logs/version_1/checkpoints/epoch=6-step=5754.ckpt"
    dataset_root = "dataset/"

    lr = 1e-3
    ##########################################################

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, num_workers=12)

    # Model
    model = Model(lr)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=20)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

