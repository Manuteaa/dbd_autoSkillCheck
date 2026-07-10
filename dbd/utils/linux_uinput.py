# linux_uinput.py
# Linux Virtual Input Device using evdev/uinput.
#
# This module creates a virtual input device via the kernel's uinput subsystem.
# Input events are injected at the kernel level, which may appear more legitimate
# than user-space tools like xdotool.
#
# NOTE: uinput devices are detectable as virtual via sysfs paths and lack of USB bus.
# This module does NOT bypass anti-cheat detection.

import time
import atexit
import sys
from typing import Optional, Dict

# Graceful import handling
try:
    import evdev
    from evdev import UInput, ecodes as e
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

# Generic device identity - no hardware impersonation
# Using a generic vendor/product ID to avoid driver conflicts
GENERIC_VENDOR_ID = 0x1234
GENERIC_PRODUCT_ID = 0x0001
GENERIC_DEVICE_NAME = "Virtual Input Device"

# Default keymap: internal key name -> evdev ecode
# Only Space and Shift are mapped by default since DBD skill checks use Space.
# Users can extend this via the custom_keymap parameter.
DEFAULT_KEYMAP: Dict[str, int] = {
    'space': e.KEY_SPACE,
    'shift': e.KEY_LEFTSHIFT,
}


class LinuxVirtualController:
    """Virtual input device controller using evdev/uinput.
    
    Args:
        custom_keymap: Optional dict mapping key names to evdev ecodes.
                      Extends the default keymap (space, shift).
    
    Example:
        # Default: only space and shift
        ctrl = LinuxVirtualController()
        ctrl.press('space')
        
        # Custom keymap for additional keys
        custom = {'a': e.KEY_A, 'd': e.KEY_D}
        ctrl = LinuxVirtualController(custom_keymap=custom)
    """
    
    def __init__(self, custom_keymap: Optional[Dict[str, int]] = None):
        self.uinput: Optional['UInput'] = None
        self.device_name = "Unknown"
        self._keymap: Dict[str, int] = {**DEFAULT_KEYMAP, **(custom_keymap or {})}
        
        if not EVDEV_AVAILABLE:
            print("[Warning] 'evdev' library not found. Falling back to pynput.")
            return

        try:
            self._create_device()
        except PermissionError:
            print("[Error] Permission denied accessing /dev/uinput.")
            print("  -> Add udev rule: echo 'KERNEL==\"uinput\", MODE=\"0660\", GROUP=\"input\"' | sudo tee /etc/udev/rules.d/99-uinput.rules")
            print("  -> Then: sudo udevadm control --reload-rules && sudo udevadm trigger")
            print("  -> Or add your user to 'input' group: sudo usermod -aG input $USER")
            print("  -> Falling back to standard pynput (less safe).")
            self.uinput = None
        except Exception as exc:
            print(f"[Error] Failed to create virtual device: {exc}")
            self.uinput = None

    def _create_device(self):
        """Create the virtual input device."""
        cap = {
            e.EV_KEY: list(self._keymap.values()),
        }

        self.uinput = UInput(
            events=cap,
            name=GENERIC_DEVICE_NAME,
            vendor=GENERIC_VENDOR_ID,
            product=GENERIC_PRODUCT_ID,
            version=0x0001,
        )
        self.device_name = GENERIC_DEVICE_NAME
        print(f"[Core] Virtual input device initialized: {self.device_name}")

    def press(self, key_code):
        """Send key DOWN event.
        
        Args:
            key_code: Key name string (e.g. 'space', 'shift', or custom keys).
        
        Raises:
            ValueError: If the key is not in the keymap.
        """
        if self.uinput:
            target = self._keymap.get(key_code)
            if target is None:
                raise ValueError(
                    f"Unknown key: {key_code!r}. "
                    f"Available keys: {list(self._keymap.keys())}. "
                    f"Pass custom_keymap to LinuxVirtualController to add more keys."
                )
            self.uinput.write(e.EV_KEY, target, 1)
            self.uinput.syn()

    def release(self, key_code):
        """Send key UP event.
        
        Args:
            key_code: Key name string.
        
        Raises:
            ValueError: If the key is not in the keymap.
        """
        if self.uinput:
            target = self._keymap.get(key_code)
            if target is None:
                raise ValueError(
                    f"Unknown key: {key_code!r}. "
                    f"Available keys: {list(self._keymap.keys())}."
                )
            self.uinput.write(e.EV_KEY, target, 0)
            self.uinput.syn()

    def is_active(self):
        """Check if the virtual device is active."""
        return self.uinput is not None

    def close(self):
        """Close the virtual device and release resources."""
        if self.uinput:
            try:
                self.uinput.close()
            except Exception:
                pass
            self.uinput = None


# Singleton instance
_vcontroller: Optional[LinuxVirtualController] = None


def get_controller(custom_keymap: Optional[Dict[str, int]] = None) -> LinuxVirtualController:
    """Get or create the singleton controller.
    
    Args:
        custom_keymap: Optional keymap for the first call (subsequent calls ignore this).
    """
    global _vcontroller
    if _vcontroller is None:
        _vcontroller = LinuxVirtualController(custom_keymap=custom_keymap)
        # Register cleanup on exit
        atexit.register(_vcontroller.close)
    return _vcontroller


def close_controller():
    """Explicitly close the singleton controller."""
    global _vcontroller
    if _vcontroller is not None:
        _vcontroller.close()
        _vcontroller = None
