import mss
import numpy as np
import onnxruntime
import pyautogui
from PIL import Image

class AI_model:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    pred_dict = {0: {"desc": "None", "hit": False},
                 1: {"desc": "repair-heal (great)", "hit": True},
                 2: {"desc": "repair-heal (ante-frontier)", "hit": True},
                 3: {"desc": "repair-heal (out)", "hit": False},
                 4: {"desc": "full white (great)", "hit": True},
                 5: {"desc": "full white (out)", "hit": False},
                 6: {"desc": "full black (great)", "hit": True},
                 7: {"desc": "full black (out)", "hit": False},
                 8: {"desc": "wiggle (great)", "hit": True},
                 9: {"desc": "wiggle (frontier)", "hit": False},
                 10: {"desc": "wiggle (out)", "hit": False}
                 }

    def __init__(self, onnx_filepath=None, use_gpu=False):
        if onnx_filepath is None:
            onnx_filepath = "model.onnx"

        if use_gpu:
            import torch  # Required to load cudnn, even if torch will not be directly used
            execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            execution_providers = ['CPUExecutionProvider']

        # Trained model
        self.ort_session = onnxruntime.InferenceSession(onnx_filepath, providers=execution_providers)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.mss = mss.mss()
        self.monitor = self._get_monitor_attributes()

    def is_using_cuda(self):
        active_providers = self.ort_session.get_providers()
        is_using_cuda = "CUDAExecutionProvider" in active_providers
        return is_using_cuda

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
        img = img / 255.0
        img = (img - self.MEAN) / self.STD

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        img = np.float32(img)
        return img

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, image):
        ort_inputs = {self.input_name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        logits = np.squeeze(ort_outs)

        pred = int(np.argmax(logits))
        probs = self.softmax(logits)

        probs = np.round(probs, decimals=3).tolist()
        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}
        should_hit = self.pred_dict[pred]["hit"]
        desc = self.pred_dict[pred]["desc"]

        return pred, desc, probs_dict, should_hit
