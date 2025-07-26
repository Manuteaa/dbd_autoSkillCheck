import numpy as np
import bettercam
from PIL import Image

from dbd.utils.monitoring_mss import Monitoring


class Monitoring_bettercam(Monitoring):
    def __enter__(self, monitor_region=None, monitor_id=0, target_fps=240):
        self.bettercam_camera = bettercam.create(max_buffer_len=1, output_color="RGB", output_idx=monitor_id)
        self.bettercam_camera.start(region=monitor_region, target_fps=target_fps)
        return self

    def __exit__(self):
        self.bettercam_camera.stop()
        self.bettercam_camera = None

    def get_frame_pil(self) -> Image:
        frame = self.get_frame_np()
        frame = Image.fromarray(frame)
        return frame

    def get_frame_np(self) -> np.ndarray:
        frame = self.bettercam_camera.get_latest_frame()
        return frame
