import numpy as np
from mss import mss
from PIL import Image


class Monitoring_mss:
    def __enter__(self):
        self.sct = mss()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sct.close()

    def get_monitors_info(self):
        monitors = self.sct.monitors[1:]
        monitor_choices = [(f"Monitor {i + 1}: {m['width']}x{m['height']}", i + 1) for i, m in enumerate(monitors)]
        return monitor_choices

    def get_monitor_region(self, monitor_id=1, crop_size=224, offset=True):
        """
        Get the region of the monitor to capture based on the monitor ID and crop size.
        The crop size region is size 224x224 centered on the monitor IF monitor resolution is 1920x1080.
        If the monitor resolution is different, the crop size is adjusted proportionally.

        Args:
            monitor_id (int): The ID of the monitor to capture.
            crop_size (int): The size of the crop (default is 224).
            offset (bool): Whether to apply monitor id offset (must be True to compute regions in mss convention).
        Returns:
            dict: (top, left, width, height)
        """

        monitor = self.sct.monitors[monitor_id]
        object_size_h_ratio = crop_size / 1080
        object_size = int(object_size_h_ratio * monitor["height"])

        region = {
                "top": monitor["height"] // 2 - object_size // 2,
                "left": monitor["width"] // 2 - object_size // 2,
                "width": object_size,
                "height": object_size
            }

        if offset:
            region["top"] += monitor["top"]
            region["left"] += monitor["left"]

        return region

    def get_frame_pil(self, monitor_region) -> Image:
        frame = self.sct.grab(monitor_region)
        frame = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
        return frame

    def get_frame_np(self, monitor_region) -> np.ndarray:
        frame = self.sct.grab(monitor_region)
        frame = np.array(frame, dtype=np.uint8)
        frame = np.flip(frame[:, :, :3], 2)
        return frame
