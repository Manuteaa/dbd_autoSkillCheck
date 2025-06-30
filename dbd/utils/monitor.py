from mss import mss
from PIL import Image


def get_monitors():
    with mss() as sct:
        monitors = sct.monitors[1:]
        monitor_choices = [(f"Monitor {i + 1}: {m['width']}x{m['height']}", i + 1) for i, m in enumerate(monitors)]

    return monitor_choices


def get_monitor_attributes(monitor_id=1, crop_size=224):
    with mss() as sct:
        monitor = sct.monitors[monitor_id]

        object_size_h_ratio = crop_size / 1080  # AI model was trained on 224x224 images on a 1920x1080 monitor reference
        object_size = int(object_size_h_ratio * monitor["height"])

        return {
            "top": monitor["top"] + monitor["height"] // 2 - object_size // 2,
            "left": monitor["left"] + monitor["width"] // 2 - object_size // 2,
            "width": object_size,
            "height": object_size
        }


def get_frame(monitor_attributes):
    with mss() as sct:
        frame = sct.grab(monitor_attributes)
        img_pil = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
        return img_pil
