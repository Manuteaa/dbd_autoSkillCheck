"""
v4l2loopback (OBS Virtual Camera) Screen Capture Module

This module captures frames from a v4l2 loopback device (e.g., OBS Virtual Camera).
This is the recommended method for Linux/Wayland users as it provides:
- Low latency capture
- Works on both X11 and Wayland
- No compositor-specific dependencies

Requirements:
- OBS Studio with Virtual Camera enabled (Start Virtual Camera in OBS)
- v4l2loopback kernel module (usually auto-installed with OBS)
- opencv-python (pip install opencv-python)

Setup:
1. Open OBS Studio
2. Add your game as a source (Window Capture or Game Capture)
3. Click "Start Virtual Camera" in OBS
4. Select "v4l2 (OBS VirtualCam)" in the skill check tool
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import os

from dbd.utils.monitoring_mss import Monitoring


def _is_v4l2loopback_device(device_path: str) -> bool:
    """Check if a v4l2 device is a loopback/virtual camera device."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device_path, '--info'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            info = result.stdout.lower()
            # Check for v4l2loopback or virtual camera indicators
            if 'v4l2 loopback' in info or 'virtual' in info:
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return False


# Check if v4l2loopback devices are available
# Only detect actual loopback/virtual camera devices, not webcams
V4L2_AVAILABLE = False
try:
    for i in range(10):
        dev_path = f"/dev/video{i}"
        if os.path.exists(dev_path):
            if _is_v4l2loopback_device(dev_path):
                V4L2_AVAILABLE = True
                break
except Exception:
    pass


class Monitoring_v4l2(Monitoring):
    """
    v4l2loopback screen capture for Linux.
    
    Captures frames from OBS Virtual Camera or other v4l2 loopback devices.
    This is the lowest-latency option for Wayland users.
    """
    
    def __init__(self, device_id=0, crop_size=224):
        super().__init__()
        self.crop_size = crop_size
        
        if isinstance(device_id, str):
            self.device_path = device_id
        else:
            self.device_path = f"/dev/video{device_id}"
        
        self.cap = None

    def start(self):
        """Open the v4l2 device for capture."""
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open v4l2 device: {self.device_path}\n"
                "Please ensure OBS 'Start Virtual Camera' is running."
            )
        
        # Read initial frame to get dimensions
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"Could not read from v4l2 device: {self.device_path}\n"
                "Please ensure OBS 'Start Virtual Camera' is active and has a source."
            )
        
        self._frame_height, self._frame_width = frame.shape[:2]
        self.monitor_region = self._calculate_center_region(
            self._frame_width, self._frame_height, self.crop_size
        )
        
        print(f"v4l2: Capturing from {self.device_path} ({self._frame_width}x{self._frame_height})")

    def stop(self):
        """Release the capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @staticmethod
    def get_monitors_info():
        """List available v4l2 video devices."""
        devices = []
        for i in range(10):
            dev_path = f"/dev/video{i}"
            if os.path.exists(dev_path):
                # Try to get device name
                name = f"Video Device {i}"
                try:
                    result = subprocess.run(
                        ['v4l2-ctl', '-d', dev_path, '--info'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                name = line.split(':')[-1].strip()
                                break
                except Exception:
                    pass
                
                devices.append((f"{name} ({dev_path})", dev_path))
        
        if not devices:
            devices = [("No v4l2 device found", "/dev/video0")]
        
        return devices

    def _calculate_center_region(self, width, height, crop_size):
        """Calculate center crop region."""
        object_size_h_ratio = crop_size / 1080
        object_size = int(object_size_h_ratio * height)
        
        left = width // 2 - object_size // 2
        top = height // 2 - object_size // 2
        
        return {
            "left": left,
            "top": top,
            "width": object_size,
            "height": object_size
        }

    def get_raw_frame(self):
        """Grab a raw frame from the device.
        
        Raises:
            RuntimeError: If the device is not started or stops returning frames.
        """
        if self.cap is None:
            raise RuntimeError("v4l2 not started. Call start() first.")
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"v4l2 device {self.device_path} stopped returning frames. "
                "Please ensure OBS Virtual Camera is still active."
            )
        
        return frame

    def get_frame_pil(self) -> Image.Image:
        """Return frame as PIL Image."""
        frame = self.get_frame_np()
        return Image.fromarray(frame)

    def get_frame_np(self) -> np.ndarray:
        """Return center-cropped frame as numpy array (RGB)."""
        frame = self.get_raw_frame()
        
        # BGR -> RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Center crop
        region = self.monitor_region
        if frame.shape[0] > region['height'] or frame.shape[1] > region['width']:
            frame = frame[
                region['top']:region['top'] + region['height'],
                region['left']:region['left'] + region['width']
            ]
        
        # Resize
        if frame.shape[:2] != (self.crop_size, self.crop_size):
            frame = cv2.resize(frame, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        
        return frame
