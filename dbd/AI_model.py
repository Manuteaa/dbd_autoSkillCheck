import mss
import numpy as np
import onnxruntime
import pyautogui
from PIL import Image

from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE


class AI_model:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        # Trained model
        filepath = "model.onnx"
        self.ort_session = onnxruntime.InferenceSession(filepath)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.mss = mss.mss()
        self.monitor = self._get_monitor_attributes()

    def _get_monitor_attributes(self):
        width, height = pyautogui.size()
        object_size = 224

        monitor = {"top": height // 2 - object_size // 2,
                   "left": width // 2 - object_size // 2,
                   "width": object_size,
                   "height": object_size}

        return monitor

    def grab_screenshot(self):
        screenshot = self.mss.grab(self.monitor)
        return screenshot

    def screenshot_to_pil(self, screenshot):
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    def pil_to_numpy(self, image_pil):
        img = np.array(image_pil, dtype=np.float32)
        img = np.flip(img[:, :, :3], 2)
        img = img / 255.0
        img = (img - self.MEAN) / self.STD

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        img = np.float32(img)
        return img

    def predict(self, image):
        ort_inputs = {self.input_name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = np.squeeze(ort_outs)
        pred = np.argmax(logits)
        return pred

    def process(self, pred):
        # 0 : no skill check

        # Simple (HEAL - REPAIR)
        # 1 : Great or good
        # 2 : frontier or slightly before great
        # 3 : fail

        # White (STRUGGLE - OVERCHARGE)
        # 4 : great = good
        # 5 : fail

        # Black (MERCILESS)
        # 6 : great = good
        # 7 : fail

        # Double (WIGGLE)
        # 8 great
        # 9 : frontier
        # 10 : good or fail

        # Hit !!
        if pred == 1 or pred == 2 or pred == 3 or pred == 5 or pred == 7 or pred == 9:
            PressKey(SPACE)
            ReleaseKey(SPACE)
            return True

        return False
