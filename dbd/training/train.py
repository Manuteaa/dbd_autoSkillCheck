import glob
import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary

from dbd.datasets.datasetLoader import get_dataloaders
from dbd.networks.model import Model

# torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    ##########################################################
    checkpoint = "./lightning_logs/version_2/checkpoints"
    dataset_root = "dataset/"

    ##########################################################
    # checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(
        dataset_root,
        num_workers=4,
        batch_size=32,
    )

    # Model
    model = Model(lr=2e-4, total_epochs=50)
    # model = Model.load_from_checkpoint(checkpoint, strict=True, lr=1e-4, total_epochs=5)

    # Print model summary
    summary = ModelSummary(model, max_depth=3)
    print(summary)

    valid = Trainer(accelerator="gpu", devices=1, logger=False)
    valid.validate(model=model, dataloaders=dataloader_val)

    # Training
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="F1/val", mode="max")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=model.total_epochs,
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # tensorboard --logdir=lightning_logs/
