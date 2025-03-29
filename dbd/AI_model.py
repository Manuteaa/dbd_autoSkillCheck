import numpy as np
from PIL import Image
from mss import mss
import onnxruntime as ort
import atexit
from pyautogui import size as pyautogui_size

try:
    import torch
    import tensorrt as trt
except ImportError:
    pass



def get_monitor_attributes():
    width, height = pyautogui_size()
    object_size_h_ratio = 224 / 1080
    object_size = int(object_size_h_ratio * height)

    return {
        "top": height // 2 - object_size // 2,
        "left": width // 2 - object_size // 2,
        "width": object_size,
        "height": object_size
    }

class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    pred_dict = {
        0: {"desc": "None", "hit": False},
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

    def __init__(self, model_path="model.onnx", use_gpu=False, nb_cpu_threads=None):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads
        self.mss = mss()
        self.monitor = get_monitor_attributes()

        self.context = None
        self.engine = None

        if model_path.endswith(".engine"):
            self.load_tensorrt()
        else:
            self.load_onnx()

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.is_tensorrt:
            del self.context
            del self.engine
            torch.cuda.empty_cache()

    def grab_screenshot(self):
        return self.mss.grab(self.monitor)

    def screenshot_to_pil(self, screenshot):
        pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        if pil_image.width != 224 or pil_image.height != 224:
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        return pil_image

    def pil_to_numpy(self, image_pil):
        img = np.asarray(image_pil, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        return np.expand_dims(img, axis=0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def load_onnx(self):
        sess_options = ort.SessionOptions()

        if not self.use_gpu and self.nb_cpu_threads is not None:
            sess_options.intra_op_num_threads = self.nb_cpu_threads
            sess_options.inter_op_num_threads = self.nb_cpu_threads

        execution_providers = ["CPUExecutionProvider"]

        if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            execution_providers.insert(0, "CUDAExecutionProvider")

        self.ort_session = ort.InferenceSession(
            self.model_path, providers=execution_providers, sess_options=sess_options
        )

        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_dtype = self.ort_session.get_inputs()[0].type
        self.is_tensorrt = False

    def load_tensorrt(self):
        self.is_tensorrt = True
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

        self.stream = torch.cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.allocate_buffers(self.engine)

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            if -1 in tensor_shape:
                raise ValueError(f"Tensor '{tensor_name}' has a dynamic shape {tensor_shape}. Set static dimensions before inference!")

            size = trt.volume(tensor_shape)
            device_mem = torch.empty(size, dtype=torch.float32, device="cuda")
            host_mem = np.empty(size, dtype=tensor_dtype)

            bindings.append(device_mem.data_ptr())

            tensor_mode = engine.get_tensor_mode(tensor_name)
            tensor_info = {'host': host_mem, 'device': device_mem, 'name': tensor_name}

            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append(tensor_info)
            else:
                outputs.append(tensor_info)

        return inputs, outputs, bindings

    def predict(self, image):
        if isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = self.pil_to_numpy(image)

        img_np = np.ascontiguousarray(img_np)

        if self.is_tensorrt:
            torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(self.stream)

            np.copyto(self.inputs[0]['host'], img_np.ravel())
            self.inputs[0]['device'].copy_(torch.tensor(self.inputs[0]['host'], dtype=torch.float32, device="cuda"))

            self.context.execute_v2(bindings=self.bindings)

            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream): 
                output_tensor = self.outputs[0]['device'].to("cpu", non_blocking=True)

            torch.cuda.current_stream().wait_stream(stream)

            self.outputs[0]['host'][:] = output_tensor.numpy()

            torch.cuda.synchronize()
            logits = np.squeeze(self.outputs[0]['host'])
        else:
            if self.input_dtype == "tensor(float)":
                img_np = img_np.astype(np.float32)
            elif self.input_dtype == "tensor(float16)":
                img_np = img_np.astype(np.float16)

            ort_inputs = {self.input_name: img_np}
            logits = np.squeeze(self.ort_session.run(None, ort_inputs))

        pred = int(np.argmax(logits))
        probs = self.softmax(logits)
        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}

        return pred, self.pred_dict[pred]["desc"], probs_dict, self.pred_dict[pred]["hit"]

    def check_provider(self):
        return "TensorRT" if self.is_tensorrt else self.ort_session.get_providers()[0]
