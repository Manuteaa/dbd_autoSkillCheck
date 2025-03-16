import numpy as np
from PIL import Image
from mss import mss
from onnxruntime import InferenceSession, SessionOptions, ExecutionMode, GraphOptimizationLevel, get_available_providers
from dbd.utils.frame_grabber import get_monitor_attributes


class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

    def __init__(self, onnx_filepath=None, use_gpu=False, nb_cpu_threads=None):
        if onnx_filepath is None:
            onnx_filepath = "model.onnx"

        sess_options = SessionOptions()

        if use_gpu:
            available_providers = get_available_providers()
            preferred_execution_providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            execution_providers = [p for p in preferred_execution_providers if p in available_providers]
            if execution_providers[0] == "CUDAExecutionProvider":
                import torch # Required to load cudnn, even if torch will not be directly used
        else:
            execution_providers = ['CPUExecutionProvider']

            if nb_cpu_threads is not None:
                # Reduce CPU overhead
                sess_options.intra_op_num_threads = nb_cpu_threads
                sess_options.inter_op_num_threads = nb_cpu_threads

        # Trained model
        self.ort_session = InferenceSession(onnx_filepath, providers=execution_providers, sess_options=sess_options)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.mss = mss()
    
    def check_provider(self):
        active_providers = self.ort_session.get_providers()
        return active_providers[0]

    def grab_screenshot(self):
        monitor = get_monitor_attributes()
        screenshot = self.mss.grab(monitor)
        return screenshot

    def screenshot_to_pil(self, screenshot):
        pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        if pil_image.width != 224 or pil_image.height != 224:
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        return pil_image

    def pil_to_numpy(self, image_pil):
        img = np.asarray(image_pil, dtype=np.float32) / 255.
        img = (img - self.MEAN) / self.STD
        img = np.transpose(img, (2, 0, 1))[None, ...]
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