"""Regression guard for capabilities excluded from the community runtime."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME_FILES = [
    ROOT / "dbd" / "app.py",
    ROOT / "dbd" / "utils" / "configurations.py",
    ROOT / "dbd" / "inference" / "AI_model.py",
    ROOT / "dbd" / "inference" / "inferer.py",
    ROOT / "dbd" / "inference" / "monitoring.py",
    ROOT / "pyproject.toml",
]


class CommunityBoundaryTests(unittest.TestCase):
    def test_restricted_runtime_paths_are_absent(self) -> None:
        source = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES).lower()
        for forbidden in (
            "is_premium",
            "bettercam",
            "tensorrt",
            "cudaexecutionprovider",
            "dmlexecutionprovider",
            "pause_hotkey",
            "run_data_collection",
            "collect_data",
            "model_gpu",
            "onnxruntime-gpu",
            "pycuda",
        ):
            self.assertNotIn(forbidden, source)

        for excluded_path in (
            ROOT / "dbd" / "data",
            ROOT / "dbd" / "deployment" / "models_benchmark.py",
            ROOT / "dbd" / "deployment" / "benchmark.toml",
            ROOT / "dbd" / "deployment" / "fix_onnx.py",
            ROOT / "dbd" / "data_collection.py",
            ROOT / "dbd" / "data_collection_realtime.py",
            ROOT / "dbd" / "inference" / "collection.py",
            ROOT / "dbd" / "utils" / "hotkeys.py",
            ROOT / "dbd" / "utils" / "monitoring_bettercam.py",
            ROOT / "Makefile",
            ROOT / "setup.py",
        ):
            self.assertFalse(excluded_path.exists())

    def test_settings_expose_only_community_values(self) -> None:
        settings = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(set(settings), {"monitor_id", "thresh", "thresh_frontier", "delay_frontier"})


if __name__ == "__main__":
    unittest.main()
