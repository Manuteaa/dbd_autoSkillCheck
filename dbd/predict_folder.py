import os
import shutil
from glob import glob
from time import time

import numpy as np
import tqdm
from PIL import Image


def infer_from_folder_ckpt(folder, checkpoint):
    import pytorch_lightning as pl
    from dbd.networks.model import Model
    from dbd.datasets.transforms import get_validation_transforms
    from dbd.datasets.datasetLoader import DBD_dataset

    # Dataset
    images = glob(os.path.join(folder, "*.*"))
    images = np.array([[image, 0] for image in images])

    test_transforms = get_validation_transforms()
    dataset = DBD_dataset(images, test_transforms)
    dataloader = dataset.get_dataloader(batch_size=128, num_workers=8)

    model = Model()
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    preds = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True, ckpt_path=checkpoint)
    preds = np.concatenate([pred.cpu().numpy() for pred in preds], axis=0)

    results = np.stack([images[:, 0], preds], axis=-1)
    return results


def infer_from_folder_onnx(folder, model_path, use_gpu=True, nb_cpu_threads=1, copy=False, move=False):
    from dbd.AI_model import AI_model

    images = sorted(glob(os.path.join(folder, "*.*")))
    ai_model = AI_model(model_path=model_path, use_gpu=use_gpu, nb_cpu_threads=nb_cpu_threads)
    print(f"Using {ai_model.check_provider()} for inference")
    ai_model.monitor.stop()

    results = []
    for image in tqdm.tqdm(images):
        img = Image.open(image).convert("RGB")

        if img.width != 320 or img.height != 320:
            img = img.resize((320, 320), Image.Resampling.BICUBIC)

        # crop to 224x224 from center
        left = (img.width - 224) // 2
        top = (img.height - 224) // 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))

        img = np.asarray(img)
        pred, _, _, _ = ai_model.predict(img)

        pred_folder = str(pred)
        if copy: shutil.copy(image, os.path.join(folder, pred_folder, os.path.basename(image)))
        if move: shutil.move(image, os.path.join(folder, pred_folder, os.path.basename(image)))

        results.append((image, pred))

    return results


if __name__ == '__main__':
    folder = "20250814-112924/"

    for idx in range(0, 12):
        pred_folder = os.path.join(folder, str(idx))
        os.makedirs(pred_folder, exist_ok=True)

    # PREDICT
    t0 = time()
    results1 = infer_from_folder_onnx(folder, "models/model.onnx", use_gpu=True, move=True)
    print(f"Model 1: {time() - t0:.2f} seconds")

    # COMPARE
    # t0 = time()
    # results1 = infer_from_folder_onnx(folder, "models/model.onnx", use_gpu=False, nb_cpu_threads=8)
    # print(f"Model 1: {time() - t0:.2f} seconds")

    # t0 = time()
    # results2 = infer_from_folder_onnx(folder, "models/model.trt", use_gpu=True)
    # print(f"Model 2: {time() - t0:.2f} seconds")

    # Compare results
    # results1 = np.array(results1)
    # results2 = np.array(results2)
    # matching_percentage = np.mean(results1[:, 1] == results2[:, 1]) * 100
    # print(f"Matching percentage: {matching_percentage:.2f}%")
