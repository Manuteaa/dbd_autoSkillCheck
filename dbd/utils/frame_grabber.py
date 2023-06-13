import pyautogui

def get_monitor_attributes():
    width, height = pyautogui.size()
    object_size_h = height // 6
    object_size_w = width // 6
    object_size = max(object_size_w, object_size_h)

    monitor = {"top": height // 2 - object_size // 2,
               "left": width // 2 - object_size // 2,
               "width": object_size,
               "height": object_size}

    return monitor

