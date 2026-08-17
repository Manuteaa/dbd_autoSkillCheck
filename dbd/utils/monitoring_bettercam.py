import cv2
import numpy as np
import bettercam
from PIL import Image

from dbd.utils.monitoring_mss import Monitoring

BETTERCAM_MONITORS = bettercam.__factory.outputs[0]


class Monitoring_bettercam(Monitoring):
    def __init__(self, monitor_id=0, crop_size=224, target_fps=200):
        super().__init__()
        self.crop_size = crop_size
        self.target_fps = target_fps
        self.target_size = (crop_size, crop_size) 

        self.monitor_region = self._get_monitor_region(monitor_id, crop_size)
        
        self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB", output_idx=monitor_id)
        
        region_w = self.monitor_region[2] - self.monitor_region[0]
        region_h = self.monitor_region[3] - self.monitor_region[1]
        
        self.needs_resize = (region_w != crop_size) or (region_h != crop_size)

    def start(self):
        self.bettercam_camera.start(region=self.monitor_region, target_fps=self.target_fps)

    def stop(self):
        if self.bettercam_camera:
            self.bettercam_camera.stop()
            self.bettercam_camera = None

    @staticmethod
    def get_monitors_info():
        monitors = BETTERCAM_MONITORS
        monitor_choices = [(f"Monitor {i}: {m.resolution}", i) for i, m in enumerate(monitors)]
        return monitor_choices

    @staticmethod
    def _get_monitor_region(monitor_id=0, crop_size=224):
        monitor = BETTERCAM_MONITORS[monitor_id]
        w, h = monitor.resolution

        object_size_h_ratio = crop_size / 1080
        object_size = int(object_size_h_ratio * h)

        left = w // 2 - object_size // 2
        top = h // 2 - object_size // 2
        
        #  (left, top, right, bottom)
        region = (left, top, left + object_size, top + object_size)

        return region

    def get_frame_pil(self) -> Image:
        frame = self.get_frame_np()
        return Image.fromarray(frame)

    def get_frame_np(self) -> np.ndarray:
        frame = self.bettercam_camera.get_latest_frame()

        if self.needs_resize:
            return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        return frame
