"""Tests for fixed community monitoring configuration."""

import unittest
from queue import Queue
from threading import Event
from unittest.mock import Mock, patch

from dbd.inference import monitoring
from dbd.utils.configurations import RunConfiguration


class MonitoringTests(unittest.TestCase):
    def test_worker_constructs_fixed_cpu_model_and_mss(self) -> None:
        events = []
        stop_event = Event()
        stop_event.set()
        fake_model = Mock()
        fake_model.check_provider.return_value = "CPUExecutionProvider"

        with (
            patch.object(monitoring, "COMMUNITY_MODEL_PATH") as model_path,
            patch.object(monitoring, "Monitoring_mss") as monitor_class,
            patch.object(monitoring, "AIModel", return_value=fake_model) as model_class,
        ):
            model_path.exists.return_value = True
            model_path.name = "model_cpu.onnx"
            model_path.__str__.return_value = "models/model_cpu.onnx"
            monitoring.real_time_monitoring(
                {1: {"threshold": 0.51}, 2: {"threshold": 0.62, "hit_delay": 5}},
                monitor_id=2,
                emit_cb=lambda *event: events.append(event),
                stop_event=stop_event,
            )

        monitor_class.assert_called_once_with(monitor_id=2, crop_size=224)
        model_class.assert_called_once_with(
            "models/model_cpu.onnx",
            nb_cpu_threads=4,
            monitoring=monitor_class.return_value,
        )
        startup = next(event for event in events if event[0] == "startup")[1]
        self.assertEqual(startup["model_name"], "model_cpu.onnx")
        self.assertEqual(startup["device"], "CPU")
        self.assertEqual(startup["threads"], 4)
        self.assertEqual(startup["monitoring"], "MSS")
        fake_model.cleanup.assert_called_once_with()

    def test_wrapper_applies_thresholds_without_pause_or_collection_callbacks(self) -> None:
        event_queue = Queue()
        configuration = RunConfiguration(monitor_id=1, threshold=70, frontier_threshold=80, frontier_delay=9)
        with patch.object(monitoring, "real_time_monitoring") as worker:
            monitoring.run_monitoring_with_callbacks(
                configuration=configuration,
                event_queue=event_queue,
                stop_event=Event(),
            )

        kwargs = worker.call_args.kwargs
        pred_dict = worker.call_args.args[0]
        self.assertEqual(kwargs["monitor_id"], 1)
        self.assertEqual(pred_dict[1]["threshold"], 0.7)
        self.assertEqual(pred_dict[2]["threshold"], 0.8)
        self.assertEqual(pred_dict[2]["hit_delay"], 9)
        self.assertEqual(event_queue.get_nowait()["type"], "finished")


if __name__ == "__main__":
    unittest.main()
