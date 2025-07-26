import numpy as np
import bettercam
from PIL import Image

from dbd.utils.monitoring_mss import Monitoring


class Monitoring_bettercam(Monitoring):
    def __init__(self, monitor_id=1, crop_size=224, target_fps=240):
        super().__init__()
        region = self._get_monitor_region(monitor_id - 1, crop_size, offset=False)
        self.monitor_region = (region["left"], region["top"], region["left"] + region["width"], region["top"] + region["height"])
        self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB", output_idx=monitor_id - 1)
        self.target_fps = target_fps

    def start(self):
        self.bettercam_camera.start(region=self.monitor_region, target_fps=self.target_fps)

    def stop(self):
        self.bettercam_camera.stop()
        self.bettercam_camera = None

    def get_frame_pil(self) -> Image:
        frame = self.get_frame_np()
        frame = Image.fromarray(frame)
        return frame

    def get_frame_np(self) -> np.ndarray:
        frame = self.bettercam_camera.get_latest_frame()
        return frame
