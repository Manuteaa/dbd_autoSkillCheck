import numpy as np
import cv2
import mss
import pyautogui
import torch

width, height = pyautogui.size()
object_size_h = height // 6
object_size_w = width // 6
object_size = max(object_size_w, object_size_h)

monitor = {"top": height // 2 - object_size // 2,
           "left": width // 2 - object_size // 2,
           "width": object_size,
           "height": object_size}

checkpoint = "./ckpt.ckpt"
model = Dbd_Model.My_Model.load_from_checkpoint(checkpoint)
model.eval()

with mss.mss() as sct:
    i = 0
    while True:
        screenshot = np.array(sct.grab(monitor))
        input = torch.from_numpy(screenshot)
        input = torch.permute(input, (2, 0, 1))[:3]
        input = Dbd_DatasetLoader.transforms_test(input.unsqueeze(0))
        pred = model(input)

        pred_id = torch.argmax(pred, -1).item()
        print(pred_id)

        cv2.imwrite("E:/temp/dataset/result/{}/{}.png".format(pred_id, i), screenshot)
        i += 1
