import mss
import os
import glob
import torch
import time
import onnxruntime
import numpy as np
from PIL import Image

from dbd.networks.model import Model
from dbd.datasets.transforms import MEAN, STD
from dbd.utils.frame_grabber import get_monitor_attributes_test
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

MEAN = np.array(MEAN, dtype=np.float32)
STD = np.array(STD, dtype=np.float32)


def screenshot_to_numpy(screenshot):
    img = np.array(screenshot, dtype=np.float32)
    img = np.flip(img[:, :, :3], 2)
    img = img / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


if __name__ == '__main__':
    # checkpoint = "./lightning_logs/mnasnet0_5/checkpoints"
    checkpoint = "./lightning_logs/mobilenet_v3/checkpoints"

    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    monitor = get_monitor_attributes_test()
    model = Model.load_from_checkpoint(checkpoint, strict=False)

    # TO ONNX
    filepath = "model.onnx"
    input_sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    model.to_onnx(filepath, input_sample, export_params=True)
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name

    iterations = 0
    start_time = time.time()
    print("FPS: 0")
    img_idx = 0

    with mss.mss() as sct:
        with torch.no_grad():
            while True:
                screenshot = sct.grab(monitor)
                img = screenshot_to_numpy(screenshot)

                ort_inputs = {input_name: img}
                ort_outs = ort_session.run(None, ort_inputs)
                pred = np.argmax(np.squeeze(ort_outs, 0))

                # Hit spacebar
                if pred == 2:
                    # PressKey(SPACE)
                    # screenshot2 = sct.grab(monitor)
                    # ReleaseKey(SPACE)

                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    img.save("dataset/{}.png".format(img_idx))
                    img_idx += 1

                # Print fps
                iterations += 1
                if time.time() - start_time >= 1:
                    elapsed_time = time.time() - start_time
                    iterations_per_second = iterations / elapsed_time
                    print("FPS: {}".format(iterations_per_second))
                    iterations = 0
                    start_time = time.time()
