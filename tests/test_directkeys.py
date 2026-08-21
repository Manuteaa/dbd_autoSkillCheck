"""Tests for platform-specific keyboard input."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

from dbd.utils import directkeys


class DirectKeysTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows-specific test")
    def test_windows_uses_send_input_for_press_and_release(self) -> None:
        input_value = Mock()
        with (
            patch.object(directkeys, "KEYBDINPUT") as key_input_class,
            patch.object(directkeys, "INPUT", return_value=input_value),
            patch.object(directkeys, "user32") as user32,
            patch.object(directkeys.ctypes, "byref", return_value="input pointer"),
            patch.object(directkeys.ctypes, "sizeof", return_value=40),
        ):
            directkeys.PressKey(directkeys.SPACE)
            directkeys.ReleaseKey(directkeys.SPACE)

        self.assertEqual(user32.SendInput.call_count, 2)
        user32.SendInput.assert_called_with(1, "input pointer", 40)
        self.assertEqual(
            key_input_class.call_args_list,
            [
                call(wVk=directkeys.SPACE),
                call(wVk=directkeys.SPACE, dwFlags=directkeys.KEYEVENTF_KEYUP),
            ],
        )

    def test_linux_uses_pynput_for_press_and_release(self) -> None:
        module_path = Path(directkeys.__file__)
        spec = importlib.util.spec_from_file_location("directkeys_linux_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        linux_directkeys = importlib.util.module_from_spec(spec)
        pynput_module = ModuleType("pynput")
        pynput_module.keyboard = SimpleNamespace(
            Controller=Mock,
            Key=SimpleNamespace(up="up", down="down", space="space"),
        )

        with (
            patch.object(sys, "platform", "linux"),
            patch.dict(sys.modules, {"pynput": pynput_module}),
        ):
            spec.loader.exec_module(linux_directkeys)

        self.assertFalse(hasattr(linux_directkeys, "user32"))
        controller = Mock()
        with patch.object(linux_directkeys, "_get_keyboard_controller", return_value=controller):
            linux_directkeys.PressKey(linux_directkeys.SPACE)
            linux_directkeys.ReleaseKey(linux_directkeys.SPACE)

        controller.press.assert_called_once_with(linux_directkeys.keyboard.Key.space)
        controller.release.assert_called_once_with(linux_directkeys.keyboard.Key.space)


if __name__ == "__main__":
    unittest.main()
