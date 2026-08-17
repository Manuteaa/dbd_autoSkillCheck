import numpy as np
import onnxruntime as ort
import ctypes
import time
from ctypes import c_void_p, c_int, POINTER, byref

from dbd.utils.monitoring_mss import Monitoring_mss

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
    from dbd.utils.monitoring_bettercam import Monitoring_bettercam
    bettercam_ok = True
    print("Info: Bettercam feature available.")
except ImportError:
    bettercam_ok = False


class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

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

    def __init__(self, model_path="model.onnx", use_gpu=False, nb_cpu_threads=None, monitor_id=1, use_bettercam=True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads

       # self.perf_frame_count = 0
       # self.perf_accumulated_time = 0

        # Screen monitoring
        self.monitor = None
        if use_bettercam and bettercam_ok:
            self.monitor = Monitoring_bettercam(monitor_id=monitor_id-1, crop_size=224, target_fps=200)
        else:
            self.monitor = Monitoring_mss(monitor_id=monitor_id, crop_size=224)

        self.monitor.start()
        self.ort_session = None
        self.input_name = None
        self.cuda_context = None
        self.engine = None
        self.context = None
        self.stream = None
        self.tensor_shapes = None
        self.bindings = None
        self.graph_exec = None  
        self.cuGraphLaunch = None
        
        # Pinned Memory Buffers
        self.host_input = None
        self.host_output = None

        if model_path.endswith(".trt") or model_path.endswith(".engine"):
            self.load_tensorrt()
        else:
            self.load_onnx()

    def grab_screenshot(self) -> np.ndarray:
        return self.monitor.get_frame_np()

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def load_onnx(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if not self.use_gpu and self.nb_cpu_threads is not None:
            sess_options.intra_op_num_threads = self.nb_cpu_threads
            sess_options.inter_op_num_threads = self.nb_cpu_threads

        if self.use_gpu:
            available_providers = ort.get_available_providers()
            providers_list = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
            execution_providers = [p for p in providers_list if p in available_providers]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(self.model_path, providers=execution_providers, sess_options=sess_options)
        self.input_name = self.ort_session.get_inputs()[0].name
        print(f"Info: ONNX Loaded. Provider: {self.ort_session.get_providers()[0]}")

    def load_tensorrt(self):
        assert self.use_gpu and trt_ok, "TensorRT requires GPU and libraries."

        cuda.init()
        device = cuda.Device(0)
        self.cuda_context = device.make_context()
        self.stream = cuda.Stream() 

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.tensor_shapes = [self.engine.get_tensor_shape(n) for n in tensor_names]
        
        # Pinned Memory Allocation
        try:
            input_vol = trt.volume(self.tensor_shapes[0])
            self.host_input = cuda.pagelocked_empty(input_vol, dtype=np.float32).reshape(self.tensor_shapes[0])
            
            output_vol = trt.volume(self.tensor_shapes[1])
            self.host_output = cuda.pagelocked_empty(output_vol, dtype=np.float32)
        except AttributeError:
            print("Warning: Pinned memory not supported. Using standard slow memory.")
            self.host_input = np.zeros(self.tensor_shapes[0], dtype=np.float32)
            self.host_output = np.zeros(trt.volume(self.tensor_shapes[1]), dtype=np.float32)

        p_input = cuda.mem_alloc(self.host_input.nbytes)
        p_output = cuda.mem_alloc(self.host_output.nbytes)

        self.context.set_tensor_address(tensor_names[0], int(p_input))
        self.context.set_tensor_address(tensor_names[1], int(p_output))

        self.bindings = [p_input, p_output]
        
        # CUDA Graph Capture
        try:
            cuda.memcpy_htod_async(self.bindings[0], self.host_input, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

            nvcuda = ctypes.windll.LoadLibrary("nvcuda.dll")
            
            cuStreamBeginCapture = nvcuda.cuStreamBeginCapture
            cuStreamBeginCapture.argtypes = [c_void_p, c_int]
            cuStreamEndCapture = nvcuda.cuStreamEndCapture
            cuStreamEndCapture.argtypes = [c_void_p, POINTER(c_void_p)]
            cuGraphInstantiate = nvcuda.cuGraphInstantiate
            cuGraphInstantiate.argtypes = [POINTER(c_void_p), c_void_p, c_void_p, c_void_p, c_void_p]
            self.cuGraphLaunch = nvcuda.cuGraphLaunch
            self.cuGraphLaunch.argtypes = [c_void_p, c_void_p]

            stream_ptr = c_void_p(int(self.stream.handle))
            
            err = cuStreamBeginCapture(stream_ptr, 0)
            if err != 0: raise Exception(f"BeginCapture Error: {err}")
            
            self.context.execute_async_v3(self.stream.handle)
            
            graph_handle = c_void_p()
            err = cuStreamEndCapture(stream_ptr, byref(graph_handle))
            if err != 0: raise Exception(f"EndCapture Error: {err}")
            
            exec_handle = c_void_p()
            err = cuGraphInstantiate(byref(exec_handle), graph_handle, None, None, 0)
            if err != 0: raise Exception(f"Instantiate Error: {err}")

            self.graph_exec = exec_handle
            print("Info: CUDA Graph captured successfully!")

        except Exception as e:
            print(f"Warning: Graph capture failed ({e}). Using standard execution.")
            self.graph_exec = None

    def _preprocess_fast(self, img_np: np.ndarray):
        img_chw = img_np.transpose(2, 0, 1)
        np.divide(img_chw, 255.0, out=self.host_input[0], dtype=np.float32)
        self.host_input[0] -= self.MEAN
        self.host_input[0] /= self.STD

    def predict(self, img_np: np.ndarray):

        #start_time = time.perf_counter()

        if self.engine:
            # Preprocess
            self._preprocess_fast(img_np)
            
            # Transfer
            cuda.memcpy_htod_async(self.bindings[0], self.host_input, self.stream)

            # Run
            if self.graph_exec:
                stream_ptr = c_void_p(int(self.stream.handle))
                self.cuGraphLaunch(self.graph_exec, stream_ptr)
            else:
                self.context.execute_async_v3(self.stream.handle)
            
            # Transfer Back
            cuda.memcpy_dtoh_async(self.host_output, self.bindings[1], self.stream)
            
            # Sync
            self.stream.synchronize()
            
            logits = self.host_output
            pred = int(np.argmax(logits))
            probs = self.softmax(logits)
            
        else:
            # ONNX
            img = np.asarray(img_np, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = (img - self.MEAN) / self.STD
            img = np.expand_dims(img, axis=0)
            
            ort_inputs = {self.input_name: img}
            output = self.ort_session.run(None, ort_inputs)
            logits = np.squeeze(output)
            pred = int(np.argmax(logits))
            probs = self.softmax(logits)

        # AI latency
        #end_time = time.perf_counter()
        #ai_time_ms = (end_time - start_time) * 1000
        
       # self.perf_accumulated_time += ai_time_ms
        # self.perf_frame_count += 1

        #if self.perf_frame_count >= 100:
          #  avg_latency = self.perf_accumulated_time / 100
            
            # 1000ms / latency = Saniyede kaç işlem yapabilirim (Theoretical Max AI FPS)
         #   potential_fps = 1000 / avg_latency if avg_latency > 0 else 0
            
           # mode = "TensorRT(Graph+Pinned)" if (self.engine and self.graph_exec) else "TensorRT" if self.engine else "ONNX"
           # print(f">>> [AI ONLY] {mode} | Latency: {avg_latency:.4f} ms | Max Potential FPS: {potential_fps:.1f}")
            
           # self.perf_frame_count = 0
          #  self.perf_accumulated_time = 0
        # ---------------------------------------

        probs_dict = {self.pred_dict[i]["desc"]: probs[i] for i in range(len(probs))}
        return pred, self.pred_dict[pred]["desc"], probs_dict, self.pred_dict[pred]["hit"]

    def check_provider(self):
        if self.engine:
            return "TensorRT (Graph)" if self.graph_exec else "TensorRT"
        return self.ort_session.get_providers()[0]

    def cleanup(self):
        if self.monitor: self.monitor.stop()
        if self.bindings:
            for b in self.bindings: b.free()
        if self.cuda_context:
            self.cuda_context.pop()
            print("Info: Cuda context released")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.cleanup()
    def __del__(self): self.cleanup()
