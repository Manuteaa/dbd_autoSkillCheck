from typing import Self

import numpy as np

from dbd.inference.inferer import InfererOnnx
from dbd.utils.monitoring_mss import Monitoring, Monitoring_mss

# Default prediction thresholds and delays
DEFAULT_THRESHOLD = 51
DEFAULT_FRONTIER_THRESHOLD = 51
DEFAULT_FRONTIER_HIT_DELAY = 5

# Prediction dictionary mapping class IDs to their properties
prediction_dict = {
    0: {"desc": "None", "hit": False, "threshold": 0.0, "hit_delay": 0},
    1: {"desc": "repair-heal (great)", "hit": True, "threshold": DEFAULT_THRESHOLD / 100.0, "hit_delay": 0},
    2: {"desc": "repair-heal (frontier)", "hit": True, "threshold": DEFAULT_FRONTIER_THRESHOLD / 100.0, "hit_delay": DEFAULT_FRONTIER_HIT_DELAY},
    3: {"desc": "repair-heal (out)", "hit": False, "threshold": 0.0, "hit_delay": 0},
    4: {"desc": "full white (great)", "hit": True, "threshold": DEFAULT_THRESHOLD / 100.0, "hit_delay": 0},
    5: {"desc": "full white (out)", "hit": False, "threshold": 0.0, "hit_delay": 0},
    6: {"desc": "full black (great)", "hit": True, "threshold": DEFAULT_THRESHOLD / 100.0, "hit_delay": 0},
    7: {"desc": "full black (out)", "hit": False, "threshold": 0.0, "hit_delay": 0},
    8: {"desc": "wiggle (great)", "hit": True, "threshold": DEFAULT_THRESHOLD / 100.0, "hit_delay": 0},
    9: {"desc": "wiggle (out)", "hit": False, "threshold": 0.0, "hit_delay": 0},
}


class AIModel:
    """Run model inference with optional screen monitoring."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str = "models/model_cpu.onnx",
        *,
        nb_cpu_threads: int | None = None,
        monitoring: Monitoring | None = None,
    ) -> None:
        """Initialize the model and screen monitor."""
        self.model_path = model_path
        self.nb_cpu_threads = nb_cpu_threads
        self.inferer = None

        # Screen monitoring
        self.monitor = monitoring if monitoring else Monitoring_mss(crop_size=224)

        try:
            self.monitor.start()
            self.inferer = InfererOnnx(model_path, nb_cpu_threads=nb_cpu_threads)
        except Exception:
            self.cleanup()
            raise

    def grab_screenshot(self) -> np.ndarray:
        """Capture an RGB screenshot as a NumPy array."""
        return self.monitor.get_frame_np()

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _preprocess_image_for_inference(self, img_np: np.ndarray) -> np.ndarray:
        """Normalize an image for model inference."""
        img = np.asarray(img_np, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) to (C,H,W) i.e. channel first format
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        return np.ascontiguousarray(np.expand_dims(img, axis=0))

    def predict(self, img_np: np.ndarray) -> tuple[int, list[float]]:
        """Predict a class and its probabilities."""
        img_np = self._preprocess_image_for_inference(img_np)
        output = self.inferer.infer(img_np)

        logits = np.squeeze(output)
        pred = int(np.argmax(logits))
        probs = self.softmax(logits)
        return pred, probs.tolist()

    def check_provider(self) -> str:
        """Return the active inference provider."""
        if isinstance(self.inferer, InfererOnnx):
            return self.inferer.get_execution_provider()
        return "Unknown"

    def cleanup(self) -> None:
        """Release monitoring and inference resources."""
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None

        if self.inferer is not None:
            self.inferer.cleanup()
            self.inferer = None

    def __enter__(self) -> Self:
        """Return this model for context management."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release resources when leaving a context."""
        self.cleanup()

    def __del__(self) -> None:
        """Release resources before object destruction."""
        self.cleanup()
