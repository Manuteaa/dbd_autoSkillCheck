import numpy as np
import cv2
import mss
import pyautogui
import torch

import Dbd_Model
import Dbd_DatasetLoader

width, height = pyautogui.size()
object_size_h = height // 6
object_size_w = width // 6
object_size = max(object_size_w, object_size_h)

monitor = {"top": height // 2 - object_size // 2,
           "left": width // 2 - object_size // 2,
           "width": object_size,
           "height": object_size}

checkpoint = "./lightning_logs/version_2/checkpoints/epoch=1-step=1316.ckpt"
model = Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
model.eval()

with mss.mss() as sct:
    i = 0
    while True:
        screenshot = np.array(sct.grab(monitor))

        # cv2.imwrite("E:/temp/dataset/vid/e{}.png".format(i), screenshot)
        # cv2.imshow('screen', screenshot)
        # cv2.waitKey(1)
        # i += 1

        input = torch.from_numpy(screenshot)
        input = torch.permute(input, (2, 0, 1))[:3]
        input = Dbd_DatasetLoader.transforms_test(input.unsqueeze(0))
        pred = model(input)

        pred_id = torch.argmax(pred, -1)
        print(pred_id.item())
