import numpy as np
import onnxruntime as ort
from mss import mss
import cv2

from dbd.utils.monitor import get_monitor_attributes

try:
    import torch
    torch_ok = True
    print("Info: torch library found.")
except ImportError:
    torch_ok = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    trt_ok = True
    print("Info: tensorRT and pycuda library found.")
except ImportError:
    trt_ok = False

try:
    import bettercam
    bettercam_ok = True
    # print(bettercam.device_info())
    # print(bettercam.output_info())
except ImportError:
    bettercam_ok = False


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

    def __init__(self, model_path="model.onnx", use_gpu=False, nb_cpu_threads=None, monitor_id=1, use_bettercam=False):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads

        self.mss = mss()
        self.monitor = get_monitor_attributes(monitor_id, crop_size=224)

        # Bettercam feature
        self.bettercam_camera = None
        if use_bettercam and bettercam_ok:
            bettercam_monitor = (self.monitor["left"],
                                 self.monitor["top"],
                                 self.monitor["left"] + self.monitor["width"],
                                 self.monitor["top"] + self.monitor["height"])

            self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB")  # TODO: monitor id
            self.bettercam_camera.start(region=bettercam_monitor, target_fps=240)

        # Onnx model
        self.ort_session = None
        self.input_name = None

        # TensorRT model
        self.cuda_context = None
        self.engine = None
        self.context = None
        self.stream = None
        self.tensor_shapes = None
        self.bindings = None

        if model_path.endswith(".trt"):
            self.load_tensorrt()
        else:
            self.load_onnx()

    def grab_screenshot(self) -> np.ndarray:
        """
        Grab a screenshot from the monitor or BetterCam camera.
        Returns:
            np.ndarray: The screenshot as a numpy array of shape (224x224x3) in RGB format.
        """

        if self.bettercam_camera is not None:
            screenshot_np = self.bettercam_camera.get_latest_frame()
        else:
            screenshot = self.mss.grab(self.monitor)
            screenshot_np = np.array(screenshot, dtype=np.uint8)
            screenshot_np = np.flip(screenshot_np[:, :, :3], 2)  # Convert BGRA to RGB

        if screenshot_np.shape[:2] != (224, 224):
            screenshot_np = cv2.resize(screenshot_np, (224, 224), interpolation=cv2.INTER_CUBIC)  # cv2 expects BGR, but resize with RGB is fine

        return screenshot_np

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def load_onnx(self):
        sess_options = ort.SessionOptions()

        if not self.use_gpu and self.nb_cpu_threads is not None:
            sess_options.intra_op_num_threads = self.nb_cpu_threads
            sess_options.inter_op_num_threads = self.nb_cpu_threads

        if self.use_gpu:
            assert torch_ok, "GPU mode requires torch lib"
            available_providers = ort.get_available_providers()
            preferred_execution_providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            execution_providers = [p for p in preferred_execution_providers if p in available_providers]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            self.model_path, providers=execution_providers, sess_options=sess_options
        )

        self.input_name = self.ort_session.get_inputs()[0].name

    def load_tensorrt(self):
        # https://github.com/NVIDIA/TensorRT/blob/HEAD/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb
        assert self.use_gpu, "TensorRT engine model requires GPU mode. Aborting."
        assert torch_ok, "TensorRT engine model requires torch lib. Aborting."
        assert trt_ok, "TensorRT engine model requires tensorrt lib. Aborting."

        cuda.init()
        device = cuda.Device(0)
        self.cuda_context = device.make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert len(tensor_names) == 2

        self.tensor_shapes = [self.engine.get_tensor_shape(n) for n in tensor_names]
        tensor_in = np.empty(self.tensor_shapes[0], dtype=np.float32)
        tensor_out = np.empty(self.tensor_shapes[1], dtype=np.float32)

        p_input = cuda.mem_alloc(1 * tensor_in.nbytes)
        p_output = cuda.mem_alloc(1 * tensor_out.nbytes)

        self.context.set_tensor_address(tensor_names[0], int(p_input))
        self.context.set_tensor_address(tensor_names[1], int(p_output))

        self.bindings = [p_input, p_output]
        self.stream = cuda.Stream()

    def _preprocess_image_for_inference(self, img_np: np.ndarray):
        img = np.asarray(img_np, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) to (C,H,W) i.e. channel first format
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def predict(self, img_np: np.ndarray):
        img_np = self._preprocess_image_for_inference(img_np)

        if self.engine:
            output = np.empty(self.tensor_shapes[1], dtype=np.float32)
            cuda.memcpy_htod_async(self.bindings[0], img_np, self.stream)  # transfer input data to device
            self.context.execute_async_v3(self.stream.handle)  # execute model
            cuda.memcpy_dtoh_async(output, self.bindings[1], self.stream)  # transfer predictions back
            self.stream.synchronize()  # synchronize threads

        else:
            ort_inputs = {self.input_name: img_np}
            output = self.ort_session.run(None, ort_inputs)

        logits = np.squeeze(output)
        pred = int(np.argmax(logits))
        probs = self.softmax(logits)
        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}

        return pred, self.pred_dict[pred]["desc"], probs_dict, self.pred_dict[pred]["hit"]

    def check_provider(self):
        return "TensorRT" if self.engine else self.ort_session.get_providers()[0]

    def cleanup(self):
        self.stream = None
        self.context = None
        self.engine = None

        if self.bindings:
            for binding in self.bindings:
                binding.free()
            self.bindings = None

        if self.cuda_context:
            self.cuda_context.pop()
            self.cuda_context = None
            print("Info: Cuda context released")

        if self.bettercam_camera:
            self.bettercam_camera.stop()
            del self.bettercam_camera
            self.bettercam_camera = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()
