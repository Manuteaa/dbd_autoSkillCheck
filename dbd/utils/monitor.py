from mss import mss
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def get_monitors() -> List[Tuple[str, int]]:
    """
    Get list of available monitors with their display information.
    
    Returns:
        List of tuples containing monitor description and ID
        
    Raises:
        RuntimeError: If no monitors are detected
    """
    try:
        with mss() as sct:
            monitors = sct.monitors[1:]  # Skip the "All in One" monitor at index 0
            if not monitors:
                raise RuntimeError("No monitors detected")
                
            monitor_choices = [
                (f"Monitor {i + 1}: {m['width']}x{m['height']}", i + 1) 
                for i, m in enumerate(monitors)
            ]
            
        logger.info(f"Detected {len(monitor_choices)} monitor(s)")
        return monitor_choices
        
    except Exception as e:
        logger.error(f"Error detecting monitors: {e}")
        raise RuntimeError(f"Failed to detect monitors: {e}")


def get_monitor_attributes(monitor_id: int = 1, crop_size: int = 224) -> Dict[str, int]:
    """
    Get monitor attributes for screen capture with center cropping.
    
    Args:
        monitor_id: ID of the monitor to get attributes for
        crop_size: Size of the center crop region
        
    Returns:
        Dictionary with monitor capture coordinates and dimensions
        
    Raises:
        ValueError: If monitor_id is invalid
        RuntimeError: If monitor access fails
    """
    try:
        with mss() as sct:
            if monitor_id < 1 or monitor_id >= len(sct.monitors):
                raise ValueError(f"Invalid monitor ID: {monitor_id}. Available: 1-{len(sct.monitors)-1}")
                
            monitor = sct.monitors[monitor_id]
            
            # Calculate crop size relative to 1080p reference
            # AI model was trained on 224x224 images from 1920x1080 monitors
            object_size_h_ratio = crop_size / 1080
            object_size = int(object_size_h_ratio * monitor["height"])
            
            # Ensure minimum crop size
            object_size = max(object_size, crop_size)
            
            # Calculate center crop coordinates
            crop_coords = {
                "top": monitor["top"] + monitor["height"] // 2 - object_size // 2,
                "left": monitor["left"] + monitor["width"] // 2 - object_size // 2,
                "width": object_size,
                "height": object_size
            }
            
            logger.debug(f"Monitor {monitor_id} crop region: {crop_coords}")
            return crop_coords
            
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error getting monitor {monitor_id} attributes: {e}")
        raise RuntimeError(f"Failed to get monitor attributes: {e}")


def get_frame(monitor_attributes: Dict[str, int]) -> Optional[Image.Image]:
    """
    Capture a frame from the specified monitor region.
    
    Args:
        monitor_attributes: Dictionary with capture coordinates and dimensions
        
    Returns:
        PIL Image of the captured frame, or None if capture fails
    """
    try:
        with mss() as sct:
            frame = sct.grab(monitor_attributes)
            img_pil = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
            logger.debug(f"Captured frame: {img_pil.size}")
            return img_pil
            
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None
