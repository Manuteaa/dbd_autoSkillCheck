"""Smoke tests for the limited Dear PyGui interface."""

import unittest
from unittest.mock import patch

from dearpygui import dearpygui as dpg

from dbd.app import AutoSkillCheckApp


class UiBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        dpg.create_context()

    def tearDown(self) -> None:
        dpg.destroy_context()

    def test_window_contains_only_community_controls(self) -> None:
        app = AutoSkillCheckApp(dpg)
        with patch("dbd.app.get_monitor_choices", return_value=[("Monitor 1", 1)]):
            app._build_window()

        item_types = {item: dpg.get_item_type(item) for item in dpg.get_all_items()}
        text_values = {
            str(dpg.get_value(item))
            for item, item_type in item_types.items()
            if item_type == "mvAppItemType::mvText"
        }
        labels = {dpg.get_item_label(item) for item in item_types}
        all_ui_text = "\n".join(text_values | labels).lower()

        self.assertIn("This is the limited community version of DBD Auto Skill Check.", text_values)
        self.assertIn("Additional AI Settings", labels)
        self.assertIn("MSS", text_values)
        self.assertTrue(dpg.does_item_exist(app.MONITOR_TAG))
        for tag in (app.THRESHOLD_TAG, app.FRONTIER_THRESHOLD_TAG, app.FRONTIER_DELAY_TAG):
            self.assertTrue(dpg.does_item_exist(tag))
        self.assertFalse(any(item_type == "mvAppItemType::mvCheckbox" for item_type in item_types.values()))
        for forbidden in ("premium", "bettercam", "pause hotkey", "data collection", "ai device", "cpu workload"):
            self.assertNotIn(forbidden, all_ui_text)


if __name__ == "__main__":
    unittest.main()
