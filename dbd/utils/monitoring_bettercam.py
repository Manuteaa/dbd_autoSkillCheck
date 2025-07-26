import numpy as np
import bettercam
from PIL import Image

from dbd.utils.monitoring_mss import Monitoring


class Monitoring_bettercam(Monitoring):
    def __init__(self, monitor_id=1, crop_size=224, target_fps=240):
        super().__init__()
        self.monitor_id = monitor_id
        self.crop_size = crop_size
        self.target_fps = target_fps

    def __enter__(self):
        self.monitor_region = self._get_monitor_region(self.monitor_id - 1, self.crop_size, offset=False)
        self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB", output_idx=self.monitor_id - 1)
        self.bettercam_camera.start(region=self.monitor_region, target_fps=self.target_fps)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.bettercam_camera.stop()
        self.bettercam_camera = None

    def get_frame_pil(self) -> Image:
        frame = self.get_frame_np()
        frame = Image.fromarray(frame)
        return frame

    def get_frame_np(self) -> np.ndarray:
        frame = self.bettercam_camera.get_latest_frame()
        return frame
