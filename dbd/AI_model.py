import numpy as np
import onnxruntime as ort
from PIL import Image
from mss import mss
from typing import Dict, Tuple, Optional, Any, Union, List
import logging
import os

from dbd.utils.monitor import get_monitor_attributes

# Configure logger for this module
logger = logging.getLogger(__name__)

try:
    import torch
    torch_ok = True
    logger.info("PyTorch library found - GPU acceleration available")
except ImportError:
    torch_ok = False
    logger.info("PyTorch library not found - GPU mode will be limited to ONNX CPU")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    trt_ok = True
    logger.info("TensorRT and PyCUDA libraries found - TensorRT optimization available")
except ImportError:
    trt_ok = False
    logger.info("TensorRT or PyCUDA library not found - TensorRT optimization unavailable")


class AI_model:
    """
    AI model class for Dead by Daylight skill check detection.
    
    Supports ONNX and TensorRT models with CPU/GPU execution.
    Handles screen capture, image preprocessing, and skill check prediction.
    """
    
    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Skill check prediction categories
    pred_dict: Dict[int, Dict[str, Union[str, bool]]] = {
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

    def __init__(self, model_path: str = "model.onnx", use_gpu: bool = False, 
                 nb_cpu_threads: Optional[int] = None, monitor_id: int = 1) -> None:
        """
        Initialize the AI model for skill check detection.
        
        Args:
            model_path: Path to the ONNX or TensorRT model file
            use_gpu: Whether to use GPU acceleration
            nb_cpu_threads: Number of CPU threads for inference (None for auto)
            monitor_id: Monitor ID to capture from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads
        
        # Validate model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Initializing AI model: {model_path}")
        logger.info(f"GPU mode: {use_gpu}, CPU threads: {nb_cpu_threads}")
        
        # Initialize screen capture
        self.mss = mss()
        self.monitor = get_monitor_attributes(monitor_id, crop_size=224)
        logger.info(f"Monitor configuration: {self.monitor}")

        # Initialize model variables
        self._init_model_variables()

        # Load appropriate model type
        if model_path.endswith(".trt"):
            self.load_tensorrt()
        else:
            self.load_onnx()
            
        logger.info("AI model initialization completed successfully")

    def _init_model_variables(self) -> None:
        """Initialize model-related instance variables."""
        # ONNX model variables
        self.ort_session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None

        # TensorRT model variables
        self.cuda_context: Optional[Any] = None
        self.engine: Optional[Any] = None
        self.context: Optional[Any] = None
        self.stream: Optional[Any] = None
        self.tensor_shapes: Optional[List[Tuple]] = None
        self.bindings: Optional[List[Any]] = None

    def grab_screenshot(self) -> Any:
        """Capture a screenshot from the configured monitor region."""
        try:
            return self.mss.grab(self.monitor)
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise

    def screenshot_to_pil(self, screenshot: Any) -> Image.Image:
        """
        Convert MSS screenshot to PIL Image with proper resizing.
        
        Args:
            screenshot: MSS screenshot object
            
        Returns:
            PIL Image resized to 224x224
        """
        try:
            pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            if pil_image.width != 224 or pil_image.height != 224:
                pil_image = pil_image.resize((224, 224), Image.Resampling.BICUBIC)
            return pil_image
        except Exception as e:
            logger.error(f"Failed to convert screenshot to PIL: {e}")
            raise

    def pil_to_numpy(self, image_pil: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to normalized numpy array for model input.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Normalized numpy array ready for model inference
        """
        try:
            img = np.asarray(image_pil, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]  # Normalize
            return np.expand_dims(img, axis=0)  # Add batch dimension
        except Exception as e:
            logger.error(f"Failed to convert PIL to numpy: {e}")
            raise

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to logits for probability distribution."""
        try:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        except Exception as e:
            logger.error(f"Failed to apply softmax: {e}")
            raise

    def load_onnx(self) -> None:
        """
        Load ONNX model with appropriate execution providers.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info("Loading ONNX model...")
            sess_options = ort.SessionOptions()

            # Configure CPU threading
            if not self.use_gpu and self.nb_cpu_threads is not None:
                sess_options.intra_op_num_threads = self.nb_cpu_threads
                sess_options.inter_op_num_threads = self.nb_cpu_threads
                logger.info(f"Configured CPU threads: {self.nb_cpu_threads}")

            # Select execution providers
            if self.use_gpu:
                if not torch_ok:
                    logger.warning("GPU mode requested but PyTorch not available, falling back to CPU")
                    execution_providers = ["CPUExecutionProvider"]
                else:
                    available_providers = ort.get_available_providers()
                    preferred_execution_providers = [
                        'CUDAExecutionProvider', 
                        'DmlExecutionProvider', 
                        'CPUExecutionProvider'
                    ]
                    execution_providers = [
                        p for p in preferred_execution_providers 
                        if p in available_providers
                    ]
                    logger.info(f"Available GPU providers: {execution_providers}")
            else:
                execution_providers = ["CPUExecutionProvider"]

            # Create inference session
            self.ort_session = ort.InferenceSession(
                self.model_path, 
                providers=execution_providers, 
                sess_options=sess_options
            )

            self.input_name = self.ort_session.get_inputs()[0].name
            
            # Log model information
            input_shape = self.ort_session.get_inputs()[0].shape
            output_shape = self.ort_session.get_outputs()[0].shape
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            logger.info(f"Using execution providers: {self.ort_session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"ONNX model loading failed: {e}")

    def load_tensorrt(self) -> None:
        """
        Load TensorRT engine for GPU inference.
        
        Raises:
            RuntimeError: If TensorRT loading fails or requirements not met
        """
        # Validate requirements
        if not self.use_gpu:
            raise RuntimeError("TensorRT engine model requires GPU mode")
        if not torch_ok:
            raise RuntimeError("TensorRT engine model requires PyTorch library")
        if not trt_ok:
            raise RuntimeError("TensorRT engine model requires TensorRT and PyCUDA libraries")

        try:
            logger.info("Loading TensorRT engine...")
            
            # Initialize CUDA
            cuda.init()
            device = cuda.Device(0)
            self.cuda_context = device.make_context()

            # Load TensorRT engine
            logger.Logger.SEVERITY = trt.Logger.Severity.WARNING
            logger_trt = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger_trt)

            with open(self.model_path, "rb") as f:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                if not self.engine:
                    raise RuntimeError("Failed to deserialize TensorRT engine")
                    
                self.context = self.engine.create_execution_context()
                if not self.context:
                    raise RuntimeError("Failed to create TensorRT execution context")

            # Setup tensor information
            tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            if len(tensor_names) != 2:
                raise RuntimeError(f"Expected 2 tensors (input + output), got {len(tensor_names)}")

            self.tensor_shapes = [self.engine.get_tensor_shape(n) for n in tensor_names]
            tensor_in = np.empty(self.tensor_shapes[0], dtype=np.float32)
            tensor_out = np.empty(self.tensor_shapes[1], dtype=np.float32)

            # Allocate GPU memory
            p_input = cuda.mem_alloc(tensor_in.nbytes)
            p_output = cuda.mem_alloc(tensor_out.nbytes)

            # Set tensor addresses
            self.context.set_tensor_address(tensor_names[0], int(p_input))
            self.context.set_tensor_address(tensor_names[1], int(p_output))

            self.bindings = [p_input, p_output]
            self.stream = cuda.Stream()
            
            logger.info(f"TensorRT engine loaded successfully")
            logger.info(f"Input shape: {self.tensor_shapes[0]}")
            logger.info(f"Output shape: {self.tensor_shapes[1]}")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise RuntimeError(f"TensorRT engine loading failed: {e}")

    def predict(self, img_np: np.ndarray) -> Tuple[int, str, Dict[str, float], bool]:
        """
        Run model inference on input image.
        
        Args:
            img_np: Preprocessed image as numpy array
            
        Returns:
            Tuple of (prediction_id, description, probabilities_dict, should_hit)
        """
        try:
            img_np = np.ascontiguousarray(img_np)

            if self.engine:
                # TensorRT inference
                output = np.empty(self.tensor_shapes[1], dtype=np.float32)
                cuda.memcpy_htod_async(self.bindings[0], img_np, self.stream)
                self.context.execute_async_v3(self.stream.handle)
                cuda.memcpy_dtoh_async(output, self.bindings[1], self.stream)
                self.stream.synchronize()
            else:
                # ONNX inference
                if not self.ort_session:
                    raise RuntimeError("No model loaded for inference")
                ort_inputs = {self.input_name: img_np}
                output = self.ort_session.run(None, ort_inputs)

            # Process predictions
            logits = np.squeeze(output)
            pred = int(np.argmax(logits))
            probs = self.softmax(logits)
            
            # Create probability dictionary
            probs_dict = {
                self.pred_dict[i]["desc"]: float(probs[i]) 
                for i in range(len(probs))
            }

            # Get prediction details
            pred_info = self.pred_dict.get(pred, {"desc": "Unknown", "hit": False})
            description = pred_info["desc"]
            should_hit = pred_info["hit"]
            
            logger.debug(f"Prediction: {pred} ({description}), confidence: {probs[pred]:.3f}")
            
            return pred, description, probs_dict, should_hit
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return safe defaults
            return 0, "Error", {"Error": 1.0}, False

    def check_provider(self) -> str:
        """Get the current execution provider name."""
        try:
            if self.engine:
                return "TensorRT"
            elif self.ort_session:
                providers = self.ort_session.get_providers()
                return providers[0] if providers else "Unknown"
            else:
                return "No model loaded"
        except Exception as e:
            logger.error(f"Failed to check provider: {e}")
            return "Error"

    def cleanup(self) -> None:
        """Clean up all model resources safely."""
        try:
            logger.info("Starting AI model cleanup...")
            
            # Clean up TensorRT resources
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.synchronize()
                    self.stream = None
                    logger.debug("TensorRT stream cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up TensorRT stream: {e}")

            if hasattr(self, 'context') and self.context:
                try:
                    del self.context
                    self.context = None
                    logger.debug("TensorRT context cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up TensorRT context: {e}")

            if hasattr(self, 'engine') and self.engine:
                try:
                    del self.engine
                    self.engine = None
                    logger.debug("TensorRT engine cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up TensorRT engine: {e}")

            # Free GPU memory bindings
            if hasattr(self, 'bindings') and self.bindings:
                try:
                    for binding in self.bindings:
                        if binding:
                            binding.free()
                    self.bindings = None
                    logger.debug("GPU memory bindings freed")
                except Exception as e:
                    logger.warning(f"Error freeing GPU memory bindings: {e}")

            # Clean up CUDA context
            if hasattr(self, 'cuda_context') and self.cuda_context:
                try:
                    self.cuda_context.pop()
                    self.cuda_context = None
                    logger.info("CUDA context released")
                except Exception as e:
                    logger.warning(f"Error releasing CUDA context: {e}")

            # Clean up ONNX session
            if hasattr(self, 'ort_session') and self.ort_session:
                try:
                    del self.ort_session
                    self.ort_session = None
                    logger.debug("ONNX session cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up ONNX session: {e}")

            # Clean up MSS
            if hasattr(self, 'mss') and self.mss:
                try:
                    self.mss.close()
                    self.mss = None
                    logger.debug("MSS screen capture cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up MSS: {e}")
                    
            logger.info("AI model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self) -> 'AI_model':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], 
                 exc_tb: Optional[Any]) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            # Use print here since logger might not be available during destruction
            print(f"Warning: Error during AI_model destruction: {e}")
