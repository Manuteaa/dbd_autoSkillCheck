"""Tests for community settings, events, and worker lifecycle."""

import unittest
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import Mock, patch

from dbd.app import AppSettings, AutoSkillCheckApp, RunConfiguration, load_settings
from dbd.inference.events import put_event


class FakeDearPyGui:
    def __init__(self) -> None:
        self.values = {AutoSkillCheckApp.STATUS_TAG: AutoSkillCheckApp.STATUS_WAITING}
        self.configurations: dict[str, dict] = {}
        self.user_data: dict[str, object] = {}

    def get_value(self, tag):
        return self.values.get(tag)

    def set_value(self, tag, value) -> None:
        self.values[tag] = value

    def configure_item(self, tag, **configuration) -> None:
        self.configurations.setdefault(tag, {}).update(configuration)

    def get_item_user_data(self, tag):
        return self.user_data.get(tag)

    def does_item_exist(self, _tag) -> bool:
        return False


class SettingsLoadingTests(unittest.TestCase):
    def test_loader_accepts_only_community_keys(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "settings.json"
            config_path.write_text(
                '{"monitor_id": 2, "thresh": 72, "use_gpu": true, "pause_hotkey": "f7", "unknown": true}',
                encoding="utf-8",
            )
            settings = load_settings(config_path)

        self.assertEqual(settings, AppSettings(monitor_id=2, thresh=72))
        for removed_name in ("use_gpu", "model_name", "nb_cpu_threads", "use_bettercam", "pause_hotkey"):
            self.assertFalse(hasattr(settings, removed_name))

    def test_non_object_settings_use_defaults(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "settings.json"
            config_path.write_text("[]", encoding="utf-8")
            settings = load_settings(config_path)
        self.assertEqual(settings, AppSettings())


class AppBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dpg = FakeDearPyGui()
        self.app = AutoSkillCheckApp(self.dpg)

    def test_read_run_configuration_contains_only_adjustable_values(self) -> None:
        self.dpg.values.update(
            {
                self.app.MONITOR_TAG: "Monitor 1",
                self.app.THRESHOLD_TAG: 73,
                self.app.FRONTIER_THRESHOLD_TAG: 81,
                self.app.FRONTIER_DELAY_TAG: 7,
            },
        )
        self.dpg.user_data[self.app.MONITOR_TAG] = {"Monitor 1": 1}
        with patch("dbd.app.COMMUNITY_MODEL_PATH") as model_path:
            model_path.is_file.return_value = True
            configuration = self.app._read_run_configuration()

        self.assertEqual(configuration, RunConfiguration(monitor_id=1, threshold=73, frontier_threshold=81, frontier_delay=7))
        for removed_name in ("model_path", "cpu_threads", "use_gpu", "use_bettercam", "collect_data", "pause_hotkey"):
            self.assertFalse(hasattr(configuration, removed_name))

    def test_telemetry_is_coalesced_and_finished_state_wins(self) -> None:
        self.app.event_queue = Queue()
        scores = [0.1] * 10
        put_event(self.app.event_queue, "prediction", {"description": "old", "scores": scores})
        put_event(self.app.event_queue, "prediction", {"description": "latest", "scores": scores})
        put_event(self.app.event_queue, "fps", 40)
        put_event(self.app.event_queue, "log", "ordered")
        put_event(self.app.event_queue, "finished")
        put_event(self.app.event_queue, "prediction", {"description": "stale", "scores": scores})

        events = self.app._drain_events()
        self.assertEqual([event["type"] for event in events], ["prediction", "fps", "log", "finished", "prediction"])
        for event in events:
            self.app._dispatch_event(event)

        self.assertEqual(self.dpg.values[self.app.DETECTION_TAG], "latest")
        self.assertEqual(self.dpg.values[self.app.STATUS_TAG], self.app.STATUS_STOPPED)

    def test_worker_can_start_stop_and_reap(self) -> None:
        self.app._append_log = Mock()
        started = Event()

        def worker_target(*, event_queue, stop_event) -> None:
            started.set()
            stop_event.wait(1)
            put_event(event_queue, "finished")

        self.assertTrue(self.app._start_worker(worker_target, {}, "started"))
        self.assertTrue(started.wait(1))
        self.dpg.values[self.app.STATUS_TAG] = self.app.STATUS_RUNNING
        self.app._stop()
        self.app.worker.join(1)
        self.app._poll_events()
        self.assertIsNone(self.app.worker)
        self.assertEqual(self.dpg.values[self.app.STATUS_TAG], self.app.STATUS_STOPPED)


if __name__ == "__main__":
    unittest.main()
