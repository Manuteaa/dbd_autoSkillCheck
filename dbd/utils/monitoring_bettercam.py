import cv2
import numpy as np
import bettercam
from PIL import Image

from dbd.utils.monitoring_mss import Monitoring
FACTORY = bettercam.__factory


class Monitoring_bettercam(Monitoring):
    def __init__(self, monitor_id=0, crop_size=224, target_fps=240):
        super().__init__()
        self.crop_size = crop_size
        self.target_fps = target_fps

        self.monitor_region = self._get_monitor_region(monitor_id, crop_size)
        self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB", output_idx=monitor_id)

    def start(self):
        self.bettercam_camera.start(region=self.monitor_region, target_fps=self.target_fps)

    def stop(self):
        self.bettercam_camera.stop()
        self.bettercam_camera = None

    @staticmethod
    def get_monitors_info():
        monitors = FACTORY.outputs[0]
        monitor_choices = [(f"Monitor {i}: {m.resolution}", i) for i, m in enumerate(monitors)]
        return monitor_choices

    @staticmethod
    def _get_monitor_region(monitor_id=0, crop_size=224):
        monitors = FACTORY.outputs[0]
        w, h = monitors[monitor_id].resolution  # Note: does not handle scaling setting

        object_size_h_ratio = crop_size / 1080
        object_size = int(object_size_h_ratio * h)

        left = w // 2 - object_size // 2
        top = h // 2 - object_size // 2
        region = (left, top, left + object_size, top + object_size)

        return region

    def get_frame_pil(self) -> Image:
        frame = self.get_frame_np()
        frame = Image.fromarray(frame)
        return frame

    def get_frame_np(self) -> np.ndarray:
        frame = self.bettercam_camera.get_latest_frame()

        if frame.shape[:2] != (self.crop_size, self.crop_size):
            frame = cv2.resize(frame, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)

        return frame
