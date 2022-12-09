import numpy as np
import time
import glob
import torch
import os
import torchvision
import tqdm
import torch.backends.cudnn

import Dbd_Model
import Dbd_DatasetLoader

print("cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled? ", torch.backends.cudnn.enabled)

dataset_0 = ["E:/temp/dbd/0bis"]
dataset_1 = ["E:/temp/dbd/1", "E:/temp/dbd/1bis"]
checkpoint = "./lightning_logs/version_4/checkpoints/epoch=1-step=1316.ckpt"

# datasets_path = dataset_0 + dataset_1
datasets_path = ["E:/temp/dbd/1bis"]
labels_idx = [0] + [1, 1]

model = Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
model.eval()
model.to(torch.device(0))

filenames = []
labels = []
for i in range(len(datasets_path)):
    directory = datasets_path[i]
    label = labels_idx[i]

    filenames_ = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
    labels_ = [label] * len(filenames_)

    filenames += filenames_
    labels += labels_

pbar = tqdm.tqdm(total=len(filenames))
t0 = time.time()

def get_pred(filename):
    image = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB).to(torch.device(0))
    input = Dbd_DatasetLoader.transforms_test(image.unsqueeze(0))
    with torch.no_grad():
        pred = model(input)

    pred_id = torch.argmax(pred, -1)
    pbar.update(1)
    return pred_id

preds = torch.stack([get_pred(f) for f in filenames]).squeeze(-1)
print(preds.cpu().numpy())

pbar.close()
print("Done in {}s".format(time.time() - t0))
