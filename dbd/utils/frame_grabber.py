import win32gui
import win32process
import psutil
import pyautogui

def find_dbd_window():
    """
    Find the Dead by Daylight window handle and position.
    Returns None if the window is not found.
    """
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(process_id)
                if 'DeadByDaylight' in process.name() or 'Dead by Daylight' in win32gui.GetWindowText(hwnd):
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append((hwnd, rect))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if windows:
        return windows[0]  # Return the first matching window
    return None

def get_monitor_attributes():
    dbd_window = find_dbd_window()
    
    if dbd_window:
        hwnd, (left, top, right, bottom) = dbd_window
        
        width = right - left
        height = bottom - top
        
        object_size_h_ratio = 224 / 1080
        object_size = int(object_size_h_ratio * height)
        
        monitor = {
            "top": top + height // 2 - object_size // 2,
            "left": left + width // 2 - object_size // 2,
            "width": object_size,
            "height": object_size
        }
        
        return monitor
    else:
        width, height = pyautogui.size()
        object_size_h_ratio = 224 / 1080
        object_size = int(object_size_h_ratio * height)

        monitor = {
            "top": height // 2 - object_size // 2,
            "left": width // 2 - object_size // 2,
            "width": object_size,
            "height": object_size
        }
        
        print("Warning: DeadByDaylight window not found. Using screen center instead.")
        return monitor

def get_monitor_attributes_test():
    dbd_window = find_dbd_window()
    
    if dbd_window:
        hwnd, (left, top, right, bottom) = dbd_window
        
        width = right - left
        height = bottom - top
        
        object_size = 224
        
        monitor = {
            "top": top + height // 2 - object_size // 2,
            "left": left + width // 2 - object_size // 2,
            "width": object_size,
            "height": object_size
        }
        
        return monitor
    else:
        width, height = pyautogui.size()
        object_size = 224

        monitor = {
            "top": height // 2 - object_size // 2,
            "left": width // 2 - object_size // 2,
            "width": object_size,
            "height": object_size
        }
        
        return monitor