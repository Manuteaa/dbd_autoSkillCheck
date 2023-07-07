import mss
import os
import glob
from PIL import Image
import torch
import time

from dbd.networks.model import Model
from dbd.datasets.transforms import get_test_transforms
from dbd.utils.frame_grabber import get_monitor_attributes_test

if __name__ == '__main__':
    checkpoint = "./lightning_logs/version_0/checkpoints"
    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    monitor = get_monitor_attributes_test()
    test_transforms = get_test_transforms()

    model = Model.load_from_checkpoint(checkpoint, strict=False)
    model.eval()

    iterations = 0
    start_time = time.time()
    print("FPS: 0")

    with mss.mss() as sct:
        with torch.no_grad():
            while True:

                screenshot = sct.grab(monitor)
                img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
                img = test_transforms(img).cuda()
                img = img.unsqueeze(0)

                pred = model(img)
                pred = torch.argmax(torch.squeeze(pred, 0)).item()
                # print(pred)

                iterations += 1
                if time.time() - start_time >= 1:
                    elapsed_time = time.time() - start_time
                    iterations_per_second = iterations / elapsed_time
                    print("FPS: {}".format(iterations_per_second))

                    # Reset counters
                    iterations = 0
                    start_time = time.time()
