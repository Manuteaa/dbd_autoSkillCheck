from glob import glob
import os
import pytorch_lightning as pl
import numpy as np
import shutil
import tqdm

from dbd.networks.model import Model
from dbd.datasets.transforms import get_validation_transforms
from dbd.datasets.datasetLoader import DBD_dataset


def infer_from_folder(folder, checkpoint):
    # Dataset
    images = glob(os.path.join(folder, "*.*"))
    images = np.array([[image, 0] for image in images])

    test_transforms = get_validation_transforms()
    dataset = DBD_dataset(images, test_transforms)
    dataloader = dataset.get_dataloader(batch_size=128, num_workers=8)

    # Model
    checkpoint = glob(os.path.join(checkpoint, "*.ckpt"))[-1]
    assert os.path.isfile(checkpoint)

    model = Model()
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    preds = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True, ckpt_path=checkpoint)
    preds = np.concatenate([pred.cpu().numpy() for pred in preds], axis=0)

    results = np.stack([images[:, 0], preds], axis=-1)
    return results


if __name__ == '__main__':
    dataset_source = "dataset/6"
    checkpoint = "./lightning_logs/version_2/checkpoints"

    assert os.path.isdir(dataset_source)
    preds = infer_from_folder(dataset_source, checkpoint)

    dataset_dest = "dataset_prediction"
    os.makedirs(os.path.join(dataset_dest, "0"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "1"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "2"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "3"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "4"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "5"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dest, "6"), exist_ok=True)

    for image, pred in tqdm.tqdm(preds, desc="Copying images"):
        pred = int(pred)

        if pred != 6:
            shutil.copy(image, os.path.join(dataset_dest, str(pred), os.path.basename(image)))
