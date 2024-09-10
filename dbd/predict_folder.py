from glob import glob
import os

import onnxruntime
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


def infer_from_folder_onnx(folder):
    # Dataset
    images = glob(os.path.join(folder, "*.*"))
    images = np.array([[image, 0] for image in images])  # give fake labels just to use our dataloader

    # dataloader (to automatically batch images and make the necessary image transformations)
    test_transforms = get_validation_transforms()
    dataset = DBD_dataset(images, test_transforms)
    dataloader = dataset.get_dataloader(batch_size=1, num_workers=8)

    # Model
    filepath = "model.onnx"
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name

    results = []
    for batch in tqdm.tqdm(dataloader, desc="Onnx inference"):
        img = batch[0].cpu().numpy()
        ort_inputs = {input_name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        pred = np.argmax(np.squeeze(ort_outs, 0))
        results.append(pred)

    results = np.stack([images[:, 0], results], axis=-1)
    return results


if __name__ == '__main__':
    dataset_source = "dataset_prediction"  # screenshots
    checkpoint = "./lightning_logs/version_1/checkpoints"

    assert os.path.isdir(dataset_source)
    # preds = infer_from_folder(dataset_source, checkpoint)
    preds = infer_from_folder_onnx(dataset_source)

    os.makedirs(os.path.join(dataset_source, "0"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "1"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "2"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "3"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "4"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "5"), exist_ok=True)
    os.makedirs(os.path.join(dataset_source, "6"), exist_ok=True)

    for image, pred in tqdm.tqdm(preds, desc="Moving images"):
        pred = int(pred)

        if pred != 0:
            # shutil.copy(image, os.path.join(dataset_dest, str(pred), os.path.basename(image)))
            shutil.move(image, os.path.join(dataset_source, str(pred), os.path.basename(image)))
