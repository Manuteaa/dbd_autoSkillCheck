import cv2
import numpy as np
from mss import mss
from PIL import Image


class Monitoring:
    def __init__(self):
        return

    def start(self):
        raise NotImplementedError("Must override start")

    def stop(self):
        raise NotImplementedError("Must override stop")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get_frame_pil(self):
        raise NotImplementedError("Must override get_frame_pil")

    def get_frame_np(self):
        raise NotImplementedError("Must override get_frame_np")

    @staticmethod
    def get_monitors_info():
        raise NotImplementedError("Must override get_monitors_info")


class Monitoring_mss(Monitoring):
    def __init__(self, monitor_id=1, crop_size=224):
        super().__init__()
        self.crop_size = crop_size
        self.monitor_region = self._get_monitor_region(monitor_id, crop_size)
        self.sct = None

    def start(self):
        self.sct = mss()

    def stop(self):
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    @staticmethod
    def get_monitors_info():
        with mss() as sct:
            monitors = sct.monitors[1:]
            monitor_choices = [(f"Monitor {i + 1}: {m['width']}x{m['height']}", i + 1) for i, m in enumerate(monitors)]
            return monitor_choices

    @staticmethod
    def _get_monitor_region(monitor_id=1, crop_size=224):
        with mss() as sct:
            monitor = sct.monitors[monitor_id]
            object_size_h_ratio = crop_size / 1080
            object_size = int(object_size_h_ratio * monitor["height"])

            region = {
                    "top": monitor["top"] + monitor["height"] // 2 - object_size // 2,
                    "left": monitor["left"] + monitor["width"] // 2 - object_size // 2,
                    "width": object_size,
                    "height": object_size
                }

            return region

    def get_raw_frame(self):
        if self.sct is None:
            raise RuntimeError("Monitoring_mss not started. Call start() before grabbing frames.")

        return self.sct.grab(self.monitor_region)

    def get_frame_pil(self) -> Image:
        frame = self.get_raw_frame()
        frame = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")

        if frame.height != self.crop_size or frame.width != self.crop_size:
            frame = frame.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)

        return frame

    def get_frame_np(self) -> np.ndarray:
        frame = self.get_raw_frame()
        frame = np.array(frame, dtype=np.uint8)
        frame = np.flip(frame[:, :, :3], 2)  # Convert BGRA to RGB

        if frame.shape[:2] != (self.crop_size, self.crop_size):
            frame = cv2.resize(frame, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)

        return frame
