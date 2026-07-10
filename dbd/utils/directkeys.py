# directkeys.py
# Cross-platform keyboard input handler
# Windows: Win32 SendInput API
# Linux/macOS: pynput library

import sys
import time

# Cross-platform input handling
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32', use_last_error=True)

    INPUT_MOUSE    = 0
    INPUT_KEYBOARD = 1
    INPUT_HARDWARE = 2

    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP       = 0x0002
    KEYEVENTF_UNICODE     = 0x0004
    KEYEVENTF_SCANCODE    = 0x0008

    MAPVK_VK_TO_VSC = 0

    # Key codes: msdn.microsoft.com/en-us/library/dd375731
    UP = 0x26
    DOWN = 0x28
    A = 0x41
    SPACE = 0x20
    SHIFT = 0x10

    wintypes.ULONG_PTR = wintypes.WPARAM

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (("dx",          wintypes.LONG),
                    ("dy",          wintypes.LONG),
                    ("mouseData",   wintypes.DWORD),
                    ("dwFlags",     wintypes.DWORD),
                    ("time",        wintypes.DWORD),
                    ("dwExtraInfo", wintypes.ULONG_PTR))

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (("wVk",         wintypes.WORD),
                    ("wScan",       wintypes.WORD),
                    ("dwFlags",     wintypes.DWORD),
                    ("time",        wintypes.DWORD),
                    ("dwExtraInfo", wintypes.ULONG_PTR))

        def __init__(self, *args, **kwds):
            super(KEYBDINPUT, self).__init__(*args, **kwds)
            if not self.dwFlags & KEYEVENTF_UNICODE:
                self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                     MAPVK_VK_TO_VSC, 0)

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = (("uMsg",    wintypes.DWORD),
                    ("wParamL", wintypes.WORD),
                    ("wParamH", wintypes.WORD))

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = (("ki", KEYBDINPUT),
                        ("mi", MOUSEINPUT),
                        ("hi", HARDWAREINPUT))
        _anonymous_ = ("_input",)
        _fields_ = (("type",   wintypes.DWORD),
                    ("_input", _INPUT))

    LPINPUT = ctypes.POINTER(INPUT)

    def _check_count(result, func, args):
        if result == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        return args

    user32.SendInput.errcheck = _check_count
    user32.SendInput.argtypes = (wintypes.UINT, LPINPUT, ctypes.c_int)

    def PressKey(hexKeyCode):
        x = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=hexKeyCode))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    def ReleaseKey(hexKeyCode):
        x = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=hexKeyCode, dwFlags=KEYEVENTF_KEYUP))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    ACTIVE_INPUT_MODE = "Windows API (Standard)"

else:
    # Linux / macOS implementation
    try:
        from pynput.keyboard import Key, Controller
        pynput_available = True
        keyboard = Controller()
        SPACE = Key.space
        SHIFT = Key.shift
        UP = Key.up
        DOWN = Key.down
        A = 'a'
    except ImportError:
        pynput_available = False
        SPACE = 'space' # safe fallback for constants
        SHIFT = 'shift'
        UP = 'up'
        DOWN = 'down'
        A = 'a'

    # Try to initialize Kernel-Level Input (Linux only)
    uinput_dev = None
    _uinput_close_fn = None
    if sys.platform == "linux":
        try:
            from dbd.utils.linux_uinput import get_controller, close_controller
            dev = get_controller()
            if dev.is_active():
                uinput_dev = dev
                _uinput_close_fn = close_controller
        except (ImportError, ValueError):
            pass

    if uinput_dev:
        # 🟢 Kernel-Level Input Mode (uinput)
        def PressKey(key_code):
            if key_code == SPACE or key_code == 'space':
                uinput_dev.press('space')
            elif key_code == SHIFT or key_code == 'shift':
                uinput_dev.press('shift')
            elif pynput_available:
                # Fallback to pynput for unmapped keys
                keyboard.press(key_code)
            # else: silently ignore unknown keys

        def ReleaseKey(key_code):
            if key_code == SPACE or key_code == 'space':
                uinput_dev.release('space')
            elif key_code == SHIFT or key_code == 'shift':
                uinput_dev.release('shift')
            elif pynput_available:
                keyboard.release(key_code)
                
    elif pynput_available:
        # 🟡 User-Level Input Mode (Fallback)
        def PressKey(key_code):
            keyboard.press(key_code)

        def ReleaseKey(key_code):
            keyboard.release(key_code)
    else:
        # 🔴 No input method available
        def PressKey(key_code):
            print(f"[Error] No keyboard input method available. Pressed: {key_code}")

        def ReleaseKey(key_code):
            pass

    # Set active mode string
    if uinput_dev:
        ACTIVE_INPUT_MODE = "Linux uinput (kernel-level)"
    elif pynput_available:
        ACTIVE_INPUT_MODE = "Linux User (pynput)"
    else:
        ACTIVE_INPUT_MODE = "None"

if __name__ == "__main__":
    print(f"Testing input on {sys.platform}")
    print(f"Mode: {ACTIVE_INPUT_MODE}")
    time.sleep(1)
    PressKey(SPACE)
    time.sleep(0.5)
    ReleaseKey(SPACE)
    print("Pressed Space")