"""Tests for the fixed community inference configuration."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from dbd.inference import inferer as inferer_module
from dbd.inference.AI_model import AIModel


class CpuInferenceTests(unittest.TestCase):
    def test_model_uses_balanced_cpu_profile_and_predicts_ten_classes(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        monitor = Mock()
        model_input = Mock()
        model_input.name = "input"
        session = Mock()
        session.get_inputs.return_value = [model_input]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.return_value = [np.arange(10, dtype=np.float32)[None, :]]

        with patch.object(inferer_module.ort, "InferenceSession", return_value=session) as session_class:
            with AIModel("mocked-model", nb_cpu_threads=4, monitoring=monitor) as model:
                prediction, probabilities = model.predict(image)
                self.assertEqual(model.check_provider(), "CPUExecutionProvider")

        session_options = session_class.call_args.kwargs["sess_options"]
        self.assertEqual(session_class.call_args.args, ("mocked-model",))
        self.assertEqual(session_class.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(session_options.execution_mode, inferer_module.ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(session_options.inter_op_num_threads, 1)
        self.assertEqual(session_options.intra_op_num_threads, 4)
        inference_input = session.run.call_args.args[1]["input"]
        self.assertEqual(inference_input.shape, (1, 3, 224, 224))

        monitor.start.assert_called_once_with()
        monitor.stop.assert_called_once_with()
        self.assertEqual(prediction, 9)
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
