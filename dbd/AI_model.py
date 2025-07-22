import numpy as np
import onnxruntime as ort
from PIL import Image
from mss import mss

from dbd.utils.monitor import get_monitor_attributes

try:
    import torch
    torch_ok = True
    print("Info: torch library found.")
except ImportError as e:
    torch_ok = False
    print("Info: torch library not found. You must use onnx AI model with CPU mode for inference.")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    trt_ok = True
    print("Info: tensorRT and pycuda library found.")
except ImportError as e:
    trt_ok = False
    print("Info: tensorRT or pycuda library not found.")

# Optional BetterCam support
try:
    import bettercam
    bettercam_ok = True
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

    def __init__(self, model_path="model.onnx", use_gpu=False, nb_cpu_threads=None, monitor_id=1, use_bettercam=False, bettercam_fps=240):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads
        self.use_bettercam = use_bettercam and bettercam_ok

        self.mss = mss()
        self.monitor = get_monitor_attributes(monitor_id, crop_size=224)

        # Only create bettercam camera if requested and available
        self.camera = bettercam.create() if self.use_bettercam else None
        self.bettercam_started = False
        self.bettercam_fps = bettercam_fps
        self.region = None  # Region for center crop

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

    def bettercam_region(self):
        # Only need to do this once
        if self.region is not None:
            return
        # Get full frame to determine size
        frame = self.camera.grab()
        if frame is None:
            raise RuntimeError("BetterCam: No initial frame.")
        height, width, _ = frame.shape
        crop_size = 224
        object_size_h_ratio = crop_size / 1080  
        object_size = int(object_size_h_ratio * height)
        left = (width // 2) - (object_size // 2)
        top = (height // 2) - (object_size // 2)
        right = left + object_size
        bottom = top + object_size
        self.region = (left, top, right, bottom)

    def grab_screenshot(self):
        if self.use_bettercam and self.camera is not None:
            if not self.bettercam_started:
                self.bettercam_region()
                self.camera.start(region=self.region, target_fps=self.bettercam_fps)
                self.bettercam_started = True
            # get the latest frame 
            frame = self.camera.get_latest_frame()
            if frame is None:
                raise RuntimeError("BetterCam: No latest frame. Try updating the screen.")
            return frame
        else:
            return self.mss.grab(self.monitor)

    def screenshot_to_pil(self, screenshot):
        if self.use_bettercam and self.camera is not None:
            pil_image = Image.fromarray(screenshot)
        else:
            pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        if pil_image.width != 224 or pil_image.height != 224:
            pil_image = pil_image.resize((224, 224), Image.Resampling.BICUBIC)
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

    def predict(self, img_np):
        img_np = np.ascontiguousarray(img_np)

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

        # Stop BetterCam if used
        if self.use_bettercam and self.camera is not None and self.bettercam_started:
            self.camera.stop()
            self.bettercam_started = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()
