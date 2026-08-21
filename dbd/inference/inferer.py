"""Provide the CPU ONNX Runtime inference backend."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Inferer(ABC):
    """Provide the common interface for model inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        nb_cpu_threads: int | None = None,
    ) -> None:
        """Initialize common inference settings."""
        self.model_path = model_path
        self.nb_cpu_threads = nb_cpu_threads

    @abstractmethod
    def infer(self, img_np: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed image."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release inference resources."""
        return


class InfererOnnx(Inferer):
    """Run inference with ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        nb_cpu_threads: int | None = None,
    ) -> None:
        """Initialize an ONNX Runtime backend."""
        super().__init__(model_path, nb_cpu_threads=nb_cpu_threads)
        self.ort_session = None
        self.input_name = None

        try:
            self._load_onnx()
        except Exception as e:
            error_message = f"Failed to load ONNX model: {e}"
            raise ValueError(error_message)

    def _load_onnx(self) -> None:
        """Configure and load the ONNX Runtime session."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # sess_options.add_session_config_entry("session.set_denormal_as_zero", "1")
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = False
        sess_options.enable_profiling = False

        if self.nb_cpu_threads is not None:
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = self.nb_cpu_threads

        execution_providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            self.model_path,
            providers=execution_providers,
            sess_options=sess_options,
        )

        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, img_np: np.ndarray) -> np.ndarray:
        """Run inference with ONNX Runtime."""
        ort_inputs = {self.input_name: img_np}
        return self.ort_session.run(None, ort_inputs)[0]

    def get_execution_provider(self) -> str:
        """Return the active ONNX Runtime execution provider."""
        return self.ort_session.get_providers()[0]
